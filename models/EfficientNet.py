# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# Adapted by Alessandro Mari

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torchvision.models.mobilenetv2 import _make_divisible
from .torch_model import BaseModel
from typing import List, Callable, Optional, Union, Tuple, Dict
import re
import math
from functools import partial
from argparse import ArgumentParser


class InvertedResidualConfig:

    def __init__(self, input_channels: int, kernel: int, expanded_channels: int, out_channels: int, use_se: bool,
                 stride: int, width_mult: float, se_ratio: float, id_skip: bool) -> None:
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(
            expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = (se_ratio is not None) and (0 < se_ratio <= 1)
        self.stride = stride
        self.se_ratio = se_ratio
        self.id_skip = id_skip

    @staticmethod
    def adjust_channels(channels: int, width_mult: float) -> int:
        return _make_divisible(channels * width_mult, 8)


class SqueezeExcitation(nn.Module):
    """Squeeze and excitation module. Scales each channel by a positive number"""

    def __init__(self, input_channels: int, squeeze_channels: int, se_ratio: float) -> None:
        super().__init__()
        squeeze_channels = _make_divisible(squeeze_channels * se_ratio, 1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.non_linear1 = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.non_linear2 = nn.Sigmoid()

    def _scale(self, input: Tensor) -> Tensor:
        scale = input.mean((2, 3), keepdim=True)
        scale = self.fc1(scale)
        scale = self.non_linear1(scale)
        scale = self.fc2(scale)
        return self.non_linear2(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


class ConvBNActivation(nn.Sequential):
    """Applies 2D convolution, normalization, activation layers"""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
        **kwargs
    ) -> None:
        padding = (kernel_size - 1) // 2 * \
            dilation if padding is None else padding
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU
        if conv_layer is None:
            conv_layer = nn.Conv2d
        super(ConvBNActivation, self).__init__(
            conv_layer(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation,
                       **kwargs),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


class InvertedResidual(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.
    Args:
        cnf (InvertedResidualConfig): configuration
        norm_layer: normalization layer
        se_layer: squeeze and excite layer
        image_size (tuple or list): [image_height, image_width].
    """

    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module],
                 se_layer: Callable[..., nn.Module] = SqueezeExcitation, image_size: Union[Tuple, List] = None):
        super().__init__()

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels and cnf.id_skip

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(ConvBNActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                                           norm_layer=norm_layer, activation_layer=activation_layer, conv_layer=Conv2d, padding=0))

        # depthwise
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                                       stride=cnf.stride, groups=cnf.expanded_channels,
                                       norm_layer=norm_layer, activation_layer=activation_layer, padding=0, conv_layer=Conv2d))
        if cnf.use_se:
            layers.append(se_layer(cnf.expanded_channels,
                          cnf.input_channels, cnf.se_ratio))

        # project
        image_size = calculate_output_image_size(image_size, cnf.stride)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer,
                                       activation_layer=nn.Identity, padding=0, conv_layer=Conv2d))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor, drop_connect_rate: float = None) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            if drop_connect_rate:
                result = drop_connect(
                    result, p=drop_connect_rate, training=self.training)
            result += input
        return result


class EfficientNet(BaseModel):
    """EfficientNet model.
    References:
        [1] Tan, M. and Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neuralnetworks. 
        In Proceedings of the 36th International Conference on Machine Learning, volume 97, pages 6105–6114. PMLR
    """

    def __init__(self,
                 lr: float = 0.005, weight_decay: float = 0, class_names: List = None, x_size: int = 128, y_size: int = 128, sgd: bool = True, input_channels=1,
                 num_classes: int = 35,
                 net_type='efficientnet-b0',
                 block_cfg: Callable[..., nn.Module] = InvertedResidualConfig,
                 **params
                 ) -> None:
        super().__init__(lr, weight_decay, num_classes, class_names, sgd, **params)
        self.save_hyperparameters()
        # to generate the model description by the LightningModule
        self.example_input_array = torch.zeros(
            1, input_channels, x_size, y_size, device=self.device)
        width_mult = params.pop('width_mult', 1.0)
        # compound scaling
        params_dict = {
            # Coefficients:   width,depth,res,dropout
            'efficientnet-b0': (1.0, 1.0, 224, 0.2),
            'efficientnet-b1': (1.0, 1.1, 240, 0.2),
            'efficientnet-b2': (1.1, 1.2, 260, 0.3),
            'efficientnet-b3': (1.2, 1.4, 300, 0.3),
            'efficientnet-b4': (1.4, 1.8, 380, 0.4),
            'efficientnet-b5': (1.6, 2.2, 456, 0.4),
            'efficientnet-b6': (1.8, 2.6, 528, 0.5),
            'efficientnet-b7': (2.0, 3.1, 600, 0.5),
            'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        }
        assert net_type in params_dict.keys()
        # corresponds to b0
        base_config = self.config()
        width, depth, _, dropout = params_dict[net_type]
        width_mult *= width
        bneck_conf = partial(block_cfg, width_mult=width_mult)
        adjust_channels = partial(
            block_cfg.adjust_channels, width_mult=width_mult)

        image_size = [x_size, y_size]
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        # Stem
        out_channels = adjust_channels(32)  # number of output channels
        self.first_layer = ConvBNActivation(
            input_channels, out_channels, 3, 2, padding=0, norm_layer=norm_layer, activation_layer=nn.SiLU, conv_layer=Conv2d)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in base_config:

            # Update block input and output filters based on depth multiplier.
            num_repeat = int(math.ceil(depth * block_args.pop("num_repeat")))
            expand_ratio = block_args.pop("expand_ratio")
            se_ratio = block_args["se_ratio"]
            cnf = bneck_conf(use_se=(se_ratio is not None)
                             and (0 < se_ratio <= 1), **block_args)

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(InvertedResidual(
                cnf, norm_layer, image_size=image_size))
            image_size = calculate_output_image_size(image_size, cnf.stride)
            if num_repeat > 1:  # modify block_args to keep same output size
                cnf.input_channels = cnf.out_channels
                cnf.expanded_channels = cnf.input_channels*expand_ratio
                cnf.stride = 1
            for _ in range(num_repeat - 1):
                self._blocks.append(InvertedResidual(
                    cnf, norm_layer, image_size=image_size))

        # Head
        in_channels = adjust_channels(
            block_args["out_channels"])  # output of final block
        out_channels = adjust_channels(1280)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self.last_layer = ConvBNActivation(
            in_channels, out_channels, 1, 1, padding=0, norm_layer=norm_layer, activation_layer=nn.SiLU, conv_layer=Conv2d)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(dropout)
        self._fc = nn.Linear(out_channels, num_classes)

        # set activation to memory efficient swish by default
        self._swish = nn.SiLU(inplace=True)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Add arguments 
            -   net_type : the type of network: either efficientnet-b0,...,efficientnet-b8 (default efficientnet-b0)
            -   width_mult: the width multiplier. Reduces or expands the number of channels (default 1.0)
        to the parser."""
        parser = super(EfficientNet, EfficientNet).add_model_specific_args(
            parent_parser)
        parser.add_argument('--net_type', default='efficientnet-b0', type=str)
        parser.add_argument('--width_mult', default=1.0, type=float)
        return parser

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """use convolution layer to extract feature .
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self.first_layer(inputs)

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = 0.2
            if drop_connect_rate:
                # scale drop connect_rate
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self.last_layer(x)

        return x

    def _forward_impl(self, inputs: torch.Tensor) -> torch.Tensor:
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        x = self.extract_features(inputs)
        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)
        return F.log_softmax(x, dim=1)

    def _decode_block_string(self, string: str) -> Dict:
        """Get a block through a string notation of arguments.
            Args:
                block_string (str): A string notation of arguments.
                                    Examples: 'r1_k3_s11_e1_i32_o16_se0.25_noskip'.
            Returns:
                dict: a dict with the config parameters
        """
        assert isinstance(string, str)

        ops = string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return {"num_repeat": int(options['r']),
                "kernel": int(options['k']),
                "stride": int(options['s'][0]),  # same stride for x and y
                "input_channels": int(options['i']),
                "expand_ratio": int(options['e']),
                "expanded_channels": int(options['e'])*int(options['i']),
                "out_channels": int(options['o']),
                "se_ratio": float(options['se']) if 'se' in options else None,
                "id_skip": ('noskip' not in string)}

    def _decode(self, string_list: List) -> List:
        """Decode a list of string notations to specify blocks inside the network.
        Args:
            string_list (list[str]): A list of strings, each string is a notation of block.
        Returns:
            blocks_args: A list of dict
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(self._decode_block_string(block_string))
        return blocks_args

    def config(self):
        """Configuration of the base model
        """
        # Blocks args for the whole model(efficientnet-b0 by default)
        # It will be modified in the construction of EfficientNet Class according to model
        blocks_args = [
            'r1_k3_s11_e1_i32_o16_se0.25',
            'r2_k3_s22_e6_i16_o24_se0.25',
            'r2_k5_s22_e6_i24_o40_se0.25',
            'r3_k3_s22_e6_i40_o80_se0.25',
            'r3_k5_s11_e6_i80_o112_se0.25',
            'r4_k5_s22_e6_i112_o192_se0.25',
            'r1_k3_s11_e6_i192_o320_se0.25',
        ]
        blocks_args = self._decode(blocks_args)
        return blocks_args


def drop_connect(inputs, p, training):
    """Drop connect.
    Args:
        input (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.
    Returns:
        output: Output after drop connection.
    """
    assert 0 <= p <= 1, 'p must be in range of [0,1]'

    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1],
                                dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output


def get_width_and_height_from_size(x):
    """Obtain height and width from x.
    Args:
        x (int, tuple or list): Data size.
    Returns:
        size: A tuple or list (H,W).
    """
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    """Calculates the output image size when using Conv2dSamePadding with a stride.
       Necessary for static padding. Thanks to mannatsingh for pointing this out.
    Args:
        input_image_size (int, tuple or list): Size of input image.
        stride (int, tuple or list): Conv2d operation's stride.
    Returns:
        output_image_size: A list [H,W].
    """
    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(
        input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


def get_same_padding_conv2d(image_size):
    """like TensorFlow's 'SAME' mode 
    Args:
        image_size (int or tuple): Size of the image.
    Returns:
        Conv2dSamePadding.
    """
    return partial(Conv2dSamePadding, image_size=image_size, bias=False)


class Conv2dSamePadding(nn.Module):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding module is calculated in construction function, then used in forward.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, image_size=None):
        super().__init__()
        self.stride = [stride]*2 if isinstance(stride, int) else stride
        self.dilation = [dilation]*2 if isinstance(dilation, int) else dilation

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(
            image_size, int) else image_size
        kh, kw = (kernel_size, kernel_size) if isinstance(
            kernel_size, int) else kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] +
                    (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] +
                    (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,
                                                pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        x = self.static_padding(x)
        x = self.conv(x)
        return x
