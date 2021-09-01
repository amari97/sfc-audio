from argparse import ArgumentParser
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

import torchvision.models as models
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.mobilenetv3 import InvertedResidualConfig

from .torch_model import BaseModel
from preprocessing.helpers import boolean_string


class AdaptedMobileNetV3(BaseModel):
    """Adapt the mobilenetv3 [1] model to accept inputs with 1 channel (add 1x1 conv2d + relu + bn at the beginning)
    References:
        [1] Howard, A., Sandler, M., Chu, G., Chen, L., Chen, B., Tan, M., Wang, W., Zhu, Y.,Pang, R., Vasudevan, V., Le, Q. V., and Adam, H. (2019).  
            Searching for MobileNetV3. arXiv:1905.02244
    """

    def __init__(self, lr:float=0.005, weight_decay:float=0, num_classes:int=35, class_names:List=None, x_size:int=128, 
                y_size:int=128, sgd:bool=True, pretrained:bool=False, **params)->None:
        super().__init__(lr, weight_decay, num_classes, class_names, sgd, **params)

        self.save_hyperparameters()
        # to generate the model description by the LightningModule
        self.example_input_array = torch.zeros(
            1, 1, x_size, y_size, device=self.device)

        # parameters from the MobileNetv3 torch implementation 
        reduce_divider = 2 if params.pop('_reduced_tail', False) else 1
        dilation = 2 if params.pop('_dilated', False) else 1

        width_mult = params.pop('width_mult', 1.0)

        # configuration
        bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
        adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider,
                       True, "HS", 2, dilation),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider,
                       96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider,
                       96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5

        self.first_layer = nn.Conv2d(1, 3, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(3)
        self.mobilenet = models.MobileNetV3(
            inverted_residual_setting, num_classes=self.hparams.num_classes, last_channel=last_channel)
        if pretrained:
            state_dict = load_state_dict_from_url(
                "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth", progress=True)
            # set the last layer to the actual value
            state_dict["classifier.3.weight"] = self.mobilenet.classifier[3].weight
            state_dict["classifier.3.bias"] = self.mobilenet.classifier[3].bias
            self.mobilenet.load_state_dict(state_dict)

    @staticmethod
    def add_model_specific_args(parent_parser:ArgumentParser)->ArgumentParser:
        """Add arguments 
            -   pretrained : if true, use pretrained weights on imagenet (default false)
            -   width_mult: the width multiplier. Reduces or expands the number of channels (default 1.0)
        to the parser."""
        parser = super(AdaptedMobileNetV3, AdaptedMobileNetV3).add_model_specific_args(
            parent_parser)
        parser.add_argument('--pretrained', default=False, type=boolean_string)
        parser.add_argument('--width_mult', default=1.0, type=float)
        return parser

    def _forward_impl(self, x:torch.Tensor)->torch.Tensor:
        x = self.bn1(F.relu(self.first_layer(x)))
        x = self.mobilenet(x)
        return F.log_softmax(x, dim=1)
