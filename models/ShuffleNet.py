from argparse import ArgumentParser
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Any, List
from torchvision.models.shufflenetv2 import InvertedResidual as invRes
from torchvision.models.mobilenetv2 import _make_divisible
from .torch_model import BaseModel
from functools import partial


class InvertedResidualConfig:
    """Configuration of the inverted residual block"""

    def __init__(self, input_channels: int, out_channels: int, stride: int, width_mult: float) -> None:
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_mult: float) -> int:
        # needs to be divisible by 2 since we split the channels by 2 in InvertedResidual
        return _make_divisible(channels * width_mult, 2)


class InvertedResidual(nn.Module):
    def __init__(
        self,
        config: InvertedResidualConfig
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.block = invRes(config.input_channels,
                            config.out_channels, config.stride)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ShuffleNetV2(BaseModel):
    """
    Adaptation of Torch ShuffleNetV2 implementation [1]
    References:
        [1] Ma, N., Zhang, X., Zheng, H.-T., and Sun, J. (2018). ShuffleNet V2: Practical Guidelines 
            for Efficient CNN Architecture Design. In Proceedings of the European Conference on Computer Vision
    """

    def __init__(
        self,
        lr: float = 0.005, weight_decay: float = 0, class_names: List = None, x_size: int = 128, y_size: int = 128, sgd: bool = True, input_channels=1,
        num_classes: int = 35,
        inverted_residual_cfg: Callable[...,
                                        nn.Module] = InvertedResidualConfig,
        **params
    ) -> None:
        super().__init__(lr, weight_decay, num_classes, class_names, sgd, **params)
        self.save_hyperparameters()
        # to generate the model description by the LightningModule
        self.example_input_array = torch.zeros(
            1, input_channels, x_size, y_size, device=self.device)
        width_mult = params.pop('width_mult', 1.0)

        # partial instantiation of the inverted residual block
        bneck_conf = partial(inverted_residual_cfg, width_mult=width_mult)
        adjust_channels = partial(
            inverted_residual_cfg.adjust_channels, width_mult=width_mult)

        stages_repeats, stages_out_channels = [
            4, 8, 4], [24, 116, 232, 464, 1024]
        self._stage_out_channels = stages_out_channels

        output_channels = self._stage_out_channels[0]
        output_channels_adj = adjust_channels(output_channels)

        # first layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels_adj,
                      3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels_adj),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(bneck_conf(
                input_channels, output_channels, 2))]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(bneck_conf(
                    output_channels, output_channels, 1)))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = adjust_channels(self._stage_out_channels[-1])
        # last layer of the feature extraction
        self.conv5 = nn.Sequential(
            nn.Conv2d(adjust_channels(input_channels),
                      output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        # classifier
        self.fc = nn.Linear(output_channels, num_classes)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Add arguments 
            -   width_mult: the width multiplier. Reduces or expands the number of channels (default 1.0)
        to the parser."""
        parser = super(ShuffleNetV2, ShuffleNetV2).add_model_specific_args(
            parent_parser)
        parser.add_argument('--width_mult', default=1.0, type=float)
        return parser

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
