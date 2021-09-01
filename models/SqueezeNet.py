from argparse import ArgumentParser
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from typing import Any, Callable, List, Optional
from torchvision.models.mobilenetv2 import _make_divisible
from .torch_model import BaseModel
from functools import partial


class FireModule(nn.Module):
    """Fire module with normalization layers"""

    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(FireModule, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(
            inplanes, squeeze_planes, kernel_size=1, bias=True)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1, bias=True)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1, bias=True)
        self.expand3x3_activation = nn.ReLU(inplace=True)

        self.squeeze_norm = norm_layer(squeeze_planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze_norm(self.squeeze(x)))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class FireConfig:
    """Configuration of a Fire Module"""

    def __init__(self,  inplanes: int, squeeze_planes: int, outplanes: int, width_mult: float) -> None:
        self.inplanes = self.adjust_channels(inplanes, width_mult)
        # don't adjust the number of squeeze channels
        self.squeeze_planes = squeeze_planes
        self.outplanes = self.adjust_channels(outplanes, width_mult)

    @staticmethod
    def adjust_channels(channels: int, width_mult: float) -> int:
        # needs to be divisible by 2 since we have the same number of 1x1 and 3x3 convolutions
        return _make_divisible(channels * width_mult, 2)


class Fire(nn.Module):
    """Fire module with skip connection"""

    def __init__(self, config: FireConfig, use_skip: bool = False) -> None:
        super(Fire, self).__init__()
        if use_skip:
            assert config.inplanes == config.outplanes
        self.use_skip = use_skip
        self.block = FireModule(config.inplanes, config.squeeze_planes, int(
            config.outplanes/2), int(config.outplanes/2))

    def forward(self, x: Tensor) -> Tensor:
        if self.use_skip:
            return x+self.block(x)
        else:
            return self.block(x)


class SqueezeNet(BaseModel):
    """SqueezeNetV2 model [1]. Following the official implementation in https://github.com/forresti/SqueezeNet with few changes:
    -   add batchnorm after the squeezing layers
    -   change the classifier to 
            AdaptiveAvgPool2d,
            Linear
    References:
        [1] Iandola, F., Han, S., Moskewicz, M., Ashraf, K., Dally, W., and Keutzer, K. (2016). 
            SqueezeNet:AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size. arXiv:1602.07360.
    """

    def __init__(
        self,
        lr: float = 0.005, weight_decay: float = 0, class_names: List = None, x_size: int = 128, y_size: int = 128, sgd: bool = True, input_channels=1,
        num_classes: int = 35,
        fire_cfg: Callable[..., nn.Module] = FireConfig,
        version: str = '1_1',
        **params
    ) -> None:
        super().__init__(lr, weight_decay, num_classes, class_names, sgd, **params)
        self.save_hyperparameters()
        # to generate the model description by the LightningModule
        self.example_input_array = torch.zeros(
            1, input_channels, x_size, y_size, device=self.device)
        width_mult = params.pop('width_mult', 1.0)

        bneck_conf = partial(fire_cfg, width_mult=width_mult)
        adjust_channels = partial(
            fire_cfg.adjust_channels, width_mult=width_mult)

        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, adjust_channels(
                    96), kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(bneck_conf(96, 16, 128)),
                Fire(bneck_conf(128, 16, 128), use_skip=False),
                Fire(bneck_conf(128, 32, 256)),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(bneck_conf(256, 32, 256), use_skip=False),
                Fire(bneck_conf(256, 48, 384)),
                Fire(bneck_conf(384, 48, 384), use_skip=False),
                Fire(bneck_conf(384, 64, 512)),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(bneck_conf(512, 64, 512), use_skip=False),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(input_channels,  adjust_channels(
                    64), kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(bneck_conf(64, 16, 128)),
                Fire(bneck_conf(128, 16, 128), use_skip=False),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(bneck_conf(128, 32, 256)),
                Fire(bneck_conf(256, 32, 256), use_skip=False),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(bneck_conf(256, 48, 384)),
                Fire(bneck_conf(384, 48, 384), use_skip=False),
                Fire(bneck_conf(384, 64, 512)),
                Fire(bneck_conf(512, 64, 512), use_skip=False),
            )
        else:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest

        final_conv = nn.Conv2d(adjust_channels(
            512), self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            # equivalent to a linear layer (since 1x1 images)
            final_conv,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Add arguments 
            -   version : the version of the model (default 1_1)
            -   width_mult: the width multiplier. Reduces or expands the number of channels (default 1.0)
        to the parser."""
        parser = super(SqueezeNet, SqueezeNet).add_model_specific_args(
            parent_parser)
        parser.add_argument('--version', default='1_1', type=str)
        parser.add_argument('--width_mult', default=1.0, type=float)
        return parser

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return F.log_softmax(torch.flatten(x, 1), dim=1)
