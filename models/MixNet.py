from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.models.mobilenetv2 import _make_divisible
from typing import Callable, List, Union
from .torch_model import BaseModel
from functools import partial
from .EfficientNet import SqueezeExcitation, ConvBNActivation

import math


NON_LINEARITY = {
    'ReLU': nn.ReLU,
    'Swish': nn.SiLU,
}


def _split_channels(num_chan, num_groups):
    """Split range(num_chan) in num_groups intervals. The first one is larger if num_chan is not a multiple of num_groups"""
    split = [num_chan // num_groups for _ in range(num_groups)]
    # add the remaining channels to the first group
    split[0] += num_chan - sum(split)
    return split


def Conv3x3Bn(in_channels, out_channels, stride, non_linear='ReLU'):
    return ConvBNActivation(in_channels, out_channels, 3, stride, norm_layer=nn.BatchNorm2d, activation_layer=NON_LINEARITY[non_linear])


def Conv1x1Bn(in_channels, out_channels, non_linear='ReLU'):
    return ConvBNActivation(in_channels, out_channels, 1, 1, norm_layer=nn.BatchNorm2d, activation_layer=NON_LINEARITY[non_linear])


class MixedConv2d(nn.Module):
    """ Mixed Grouped Convolution
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, List] = 3,
                 stride: int = 1, padding: int = 0, dilation: int = 1, depthwise: bool = False) -> None:
        # depthwise=True -> depthwise convolution = grouped convolution with nb group=input channels
        super(MixedConv2d, self).__init__()

        num_groups = len(kernel_size)

        kernel_size = kernel_size if isinstance(
            kernel_size, list) else [kernel_size]
        in_splits = _split_channels(in_channels, num_groups)
        if out_channels != in_channels:
            out_splits = _split_channels(out_channels, num_groups)
        else:
            out_splits = in_splits
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)

        self.grouped_conv = nn.ModuleList()
        for ks, in_ch, out_ch in zip(kernel_size, in_splits, out_splits):
            group_conv = in_ch if depthwise else 1
            padding = (ks - 1) // 2 * dilation if depthwise else padding
            self.grouped_conv.append(
                nn.Conv2d(
                    in_ch, out_ch, ks, stride=stride,
                    padding=padding, dilation=dilation, bias=False, groups=group_conv)
            )
        self.splits = in_splits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_split = torch.split(x, self.splits, 1)
        x_out = [c(x_split[i])
                 for i, c in enumerate(self.grouped_conv)]
        x = torch.cat(x_out, dim=1)
        return x


class _MixNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: List = [3],
        expand_ksize: List = [1],
        project_ksize: List = [1],
        stride: int = 1,
        expand_ratio: float = 1,
        non_linear: str = 'ReLU',
        se_ratio: float = 0.0
    ) -> None:

        super(_MixNetBlock, self).__init__()
        assert non_linear in NON_LINEARITY.keys()
        expand = (expand_ratio != 1)
        expand_channels = int(in_channels * expand_ratio)
        se = (se_ratio != 0.0)
        self.residual_connection = (
            stride == 1 and in_channels == out_channels)

        conv = []

        if expand:
            # expansion phase
            pw_expansion = ConvBNActivation(in_channels, expand_channels, expand_ksize,
                                            norm_layer=nn.BatchNorm2d, activation_layer=NON_LINEARITY[non_linear], conv_layer=MixedConv2d, padding=0)
            conv.append(pw_expansion)

        # depthwise convolution phase
        dw = ConvBNActivation(expand_channels, expand_channels, kernel_size, stride=stride,
                              norm_layer=nn.BatchNorm2d, activation_layer=NON_LINEARITY[non_linear], depthwise=True, conv_layer=MixedConv2d, padding=0)
        conv.append(dw)

        if se:
            # squeeze and excite
            squeeze_excite = SqueezeExcitation(
                expand_channels, in_channels, se_ratio)
            conv.append(squeeze_excite)

        # projection phase
        pw_projection = ConvBNActivation(expand_channels, out_channels, project_ksize,
                                         norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity, depthwise=False, conv_layer=MixedConv2d, padding=0)
        conv.append(pw_projection)

        self.conv = nn.Sequential(*conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual_connection:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MixetNetConfig:
    """Mixnet Block Configuration class"""

    def __init__(self,  in_channels: int, out_channels: int,
                 kernel_size: List, expand_ksize: List, project_ksize: List,
                 stride: int = 1, expand_ratio: int = 1,
                 non_linear: str = 'ReLU',
                 se_ratio: float = 0.0, width_mult: float = 1.0) -> None:
        self.in_channels = self.adjust_channels(in_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.expand_ksize = expand_ksize
        self.project_ksize = project_ksize
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.non_linear = non_linear
        self.se_ratio = se_ratio

    @staticmethod
    def adjust_channels(channels: int, width_mult: float) -> int:
        return _make_divisible(channels * width_mult, 8)


class MixNetBlock(nn.Module):
    def __init__(
        self,
        config: MixetNetConfig
    ) -> None:
        super(MixNetBlock, self).__init__()
        self.block = _MixNetBlock(config.in_channels, config.out_channels,
                                  config.kernel_size, config.expand_ksize, config.project_ksize,
                                  config.stride, config.expand_ratio, config.non_linear, config.se_ratio)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class MixNet(BaseModel):
    """Mixnet model [1]
    References:
        [1] Tan, M. and Le, Q. V. (2019).   MixConv:  Mixed Depthwise Convolutional Kernels. arXiv:1907.09595"""

    def __init__(self,
                 lr: float = 0.005, weight_decay: float = 0, class_names: List = None, x_size: int = 128, y_size: int = 128, sgd: bool = True, input_channels=1,
                 num_classes: int = 35,
                 net_type='mixnet_s',
                 mixed_cfg: Callable[..., nn.Module] = MixetNetConfig,
                 **params
                 ) -> None:
        super().__init__(lr, weight_decay, num_classes, class_names, sgd, **params)
        self.save_hyperparameters()
        # to generate the model description by the LightningModule
        self.example_input_array = torch.zeros(
            1, input_channels, x_size, y_size, device=self.device)
        width_mult = params.pop('width_mult', 1.0)

        # partial construction of the block
        bneck_conf = partial(mixed_cfg, width_mult=width_mult)
        adjust_channels = partial(
            mixed_cfg.adjust_channels, width_mult=width_mult)
        feature_size = adjust_channels(1536)

        # build network
        if net_type == 'mixnet_s':
            #  [in_channels, out_channels, kernel_size, expand_ksize, project_ksize, stride, expand_ratio, non_linear, se_ratio]
            config = [bneck_conf(16,  16,  [3],              [1],    [1],    1, 1, 'ReLU', 0.0),
                      bneck_conf(16,  24,  [3],              [
                                 1, 1], [1, 1], 2, 6, 'ReLU',  0.0),
                      bneck_conf(24,  24,  [3],              [
                                 1, 1], [1, 1], 1, 3, 'ReLU',  0.0),
                      bneck_conf(24,  40,  [3, 5, 7],        [
                                 1],    [1],    2, 6, 'Swish', 0.5),
                      bneck_conf(40,  40,  [3, 5],           [
                                 1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                      bneck_conf(40,  40,  [3, 5],           [
                                 1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                      bneck_conf(40,  40,  [3, 5],           [
                                 1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                      bneck_conf(40,  80,  [3, 5, 7],        [
                                 1],    [1, 1], 2, 6, 'Swish', 0.25),
                      bneck_conf(80,  80,  [3, 5],           [
                                 1],    [1, 1], 1, 6, 'Swish', 0.25),
                      bneck_conf(80,  80,  [3, 5],           [
                                 1],    [1, 1], 1, 6, 'Swish', 0.25),
                      bneck_conf(80,  120, [3, 5, 7],        [
                                 1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                      bneck_conf(120, 120, [3, 5, 7, 9],     [
                                 1, 1], [1, 1], 1, 3, 'Swish', 0.5),
                      bneck_conf(120, 120, [3, 5, 7, 9],     [
                                 1, 1], [1, 1], 1, 3, 'Swish', 0.5),
                      bneck_conf(120, 200, [3, 5, 7, 9, 11], [
                                 1],    [1],    2, 6, 'Swish', 0.5),
                      bneck_conf(200, 200, [3, 5, 7, 9],     [
                                 1],    [1, 1], 1, 6, 'Swish', 0.5),
                      bneck_conf(200, 200, [3, 5, 7, 9],     [1],    [1, 1], 1, 6, 'Swish', 0.5)]
            stem_channels = adjust_channels(16)
            dropout_rate = 0.2
        elif net_type == 'mixnet_m' or net_type == 'mixnet_l':
            config = [bneck_conf(24,  24,  [3],          [1],    [1],    1, 1, 'ReLU',  0.0),
                      bneck_conf(24,  32,  [3, 5, 7],    [1, 1], [
                                 1, 1], 2, 6, 'ReLU',  0.0),
                      bneck_conf(32,  32,  [3],          [1, 1], [
                                 1, 1], 1, 3, 'ReLU',  0.0),
                      bneck_conf(32,  40,  [3, 5, 7, 9], [1],    [
                                 1],    2, 6, 'Swish', 0.5),
                      bneck_conf(40,  40,  [3, 5],       [1, 1], [
                                 1, 1], 1, 6, 'Swish', 0.5),
                      bneck_conf(40,  40,  [3, 5],       [1, 1], [
                                 1, 1], 1, 6, 'Swish', 0.5),
                      bneck_conf(40,  40,  [3, 5],       [1, 1], [
                                 1, 1], 1, 6, 'Swish', 0.5),
                      bneck_conf(40,  80,  [3, 5, 7],    [1],    [
                                 1],    2, 6, 'Swish', 0.25),
                      bneck_conf(80,  80,  [3, 5, 7, 9], [1, 1], [
                                 1, 1], 1, 6, 'Swish', 0.25),
                      bneck_conf(80,  80,  [3, 5, 7, 9], [1, 1], [
                                 1, 1], 1, 6, 'Swish', 0.25),
                      bneck_conf(80,  80,  [3, 5, 7, 9], [1, 1], [
                                 1, 1], 1, 6, 'Swish', 0.25),
                      bneck_conf(80,  120, [3],          [1],    [
                                 1],    1, 6, 'Swish', 0.5),
                      bneck_conf(120, 120, [3, 5, 7, 9], [1, 1], [
                                 1, 1], 1, 3, 'Swish', 0.5),
                      bneck_conf(120, 120, [3, 5, 7, 9], [1, 1], [
                                 1, 1], 1, 3, 'Swish', 0.5),
                      bneck_conf(120, 120, [3, 5, 7, 9], [1, 1], [
                                 1, 1], 1, 3, 'Swish', 0.5),
                      bneck_conf(120, 200, [3, 5, 7, 9], [1],    [
                                 1],    2, 6, 'Swish', 0.5),
                      bneck_conf(200, 200, [3, 5, 7, 9], [1],    [
                                 1, 1], 1, 6, 'Swish', 0.5),
                      bneck_conf(200, 200, [3, 5, 7, 9], [1],    [
                                 1, 1], 1, 6, 'Swish', 0.5),
                      bneck_conf(200, 200, [3, 5, 7, 9], [1],    [1, 1], 1, 6, 'Swish', 0.5)]
            stem_channels = adjust_channels(24)
            dropout_rate = 0.25
            if net_type == "mixnet_l":
                width_mult *= 1.3
        else:
            raise TypeError('Unsupported MixNet type')

        # stem convolution
        self.stem_conv = Conv3x3Bn(input_channels, stem_channels, 2)

        # building MixNet blocks
        layers = []
        for cfg in config:
            layers.append(MixNetBlock(cfg))
        self.layers = nn.Sequential(*layers)

        # last several layers
        self.head_conv = Conv1x1Bn(config[-1].out_channels, feature_size)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout_rate),
                                        nn.Linear(feature_size, num_classes))

        self._initialize_weights()

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Add arguments 
            -   net_type : the type of network: either mixnet_s,mixnet_m,mixnet_l (default mixnet_s)
            -   width_mult: the width multiplier. Reduces or expands the number of channels (default 1.0)
        to the parser."""
        parser = super(MixNet, MixNet).add_model_specific_args(parent_parser)
        parser.add_argument('--net_type', default='mixnet_s', type=str)
        parser.add_argument('--width_mult', default=1.0, type=float)
        return parser

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.head_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
