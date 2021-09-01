from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F

from .torch_model import BaseModel
from typing import List
import os

from preprocessing.helpers import boolean_string
from pathlib import Path


class Res8(BaseModel):
    """
    Res8 model [1]. Following Honk implementation [2] with few changes:
        - The pooling area is set to (4,4) instead of (4,3)
        - The dilation is increasing as in res15
        - Add prelu activation func for the classifier
    Reference:
        [1] Tang, R. and Lin, J. (2018). Deep Residual Learning for Small-Footprint Keyword Spotting. arXiv:1710.10361
        [2] https://github.com/castorini/honk/blob/master/utils/model.py
    """

    def __init__(self,  lr: float = 0.005, weight_decay: float = 0, num_classes: int = 35, class_names: List = None, x_size: int = 128, y_size: int = 128, sgd: bool = True,
                 input_channels: int = 1, channels: int = 45, dilation: bool = False, pretrained: bool = False, **params) -> None:
        super().__init__(lr, weight_decay, num_classes, class_names, sgd, **params)
        self.save_hyperparameters()
        # to generate the model description by the LightningModule
        self.example_input_array = torch.zeros(
            1, input_channels, x_size, y_size, device=self.device)
        width_mult = params.pop('width_mult', 1.0)
        channels = int(width_mult*channels)

        # first layer
        self.conv0 = nn.Conv2d(input_channels, channels,
                               (3, 3), padding=(1, 1), bias=False)
        self.pool = nn.AvgPool2d((4, 4))

        self.n_layers = n_layers = 6

        if self.hparams.dilation:
            # if dilation is true increase the dilation
            self.convs = [nn.Conv2d(channels, channels, (3, 3), padding=int(2 ** (i // 2)), dilation=int(2 ** (i // 2)),
                                    bias=False) for i in range(n_layers)]
        else:
            # following the implementation of res15 (increasing dilation)
            self.convs = [nn.Conv2d(channels, channels, (3, 3), padding=int(2 ** (i // 3)), dilation=int(2 ** (i // 3)),
                                    bias=False) for i in range(n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module(
                f'bn{i + 1}', nn.BatchNorm2d(channels, affine=False))
            self.add_module(f'conv{i + 1}', conv)

        # classifier
        self.classifier = nn.Sequential(
            nn.PReLU(), nn.Linear(channels, num_classes))

        if pretrained:
            checkpoint_path = os.path.join("pretrained", "imagenet", "res8")
            files = []
            for f in Path(checkpoint_path).rglob('*.ckpt'):
                files.append(str(f))
            if len(files) > 1:
                print("Take the one with the best accuracy")
                checkpoint_path = sorted(files, key=lambda x: float(
                    x.split("val_accuracy=")[1][:4]), reverse=True)[0]
            else:
                if len(files) == 0:
                    raise ValueError("Run pretrained_imagenet.py first")
                checkpoint_path = files[0]
            state_dict = torch.load(
                checkpoint_path, map_location=lambda storage, loc: storage)['state_dict']
            # set the last layer to the actual value and change the value of the first layer
            state_dict["classifier.1.weight"] = self.classifier[1].weight
            state_dict["classifier.1.bias"] = self.classifier[1].bias
            state_dict["conv0.weight"] = self.conv0.weight
            self.load_state_dict(state_dict)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Add arguments 
            -   channels : the number of channels to use for the hidden layers (default 45)
            -   width_mult: the width multiplier. Reduces or expands the number of channels (default 1.0)
            -   dilation : if true, increase the dilation of the convolutions (default false)
            -   pretrained : if true, loads pretrained weights on (tiny) ImageNet (default false)
        to the parser."""
        parser = super(Res8, Res8).add_model_specific_args(parent_parser)
        parser.add_argument('--channels', default=45, type=int)
        parser.add_argument('--dilation', default=False, type=boolean_string)
        parser.add_argument('--pretrained', default=False, type=boolean_string)
        parser.add_argument('--width_mult', default=1.0, type=float)
        return parser

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, f'conv{i}')(x))
            if i == 0:
                if hasattr(self, 'pool'):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, f'bn{i}')(x)
        x = x.view(x.size(0), x.size(1), -1)  # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        return F.log_softmax(self.classifier(x), dim=-1)
