import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .representation import Curve, compute_mfcc
from .helpers import label_to_index, apply
from typing import Callable, List, Dict, Tuple, Union


def space_filling_curve(x: torch.Tensor, y: List[str], labels: List[str], curve: Curve, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """"
    Represent data using a space filling curve + convert labels to int
    Args:
        x (torch tensor): audio sequence of size (Nxlength), where N is the number of audio waveforms
        y (list(str)): list of associated labels
        labels (list(str)): list of labels 
        curve: pre-computed space filling curve
    Returns:
        Two torch.tensor objects:
            - a N x H x W tensor containing the image representation of each example
            - a tensor of size N with the label (=indices) of each example
    """
    length = len(curve.X)
    # padding with 0's (if necessary)
    if x.shape[-1] != length:
        x = apply(padding, x, target_length=length)

    # side length of the image
    side_length = int(np.sqrt(length))
    # batch size
    num_samples = x.shape[0]
    image = np.zeros((num_samples, side_length, side_length))

    # convert to image
    image[:, [curve.X], [curve.Y]] = np.expand_dims(x, 1)

    # convert labels to int
    index = label_to_index(y, labels)
    return torch.Tensor(image), torch.tensor(index)


def prepare_mfcc(x: torch.Tensor, y: List[str], labels: List[str], subset: str, center: bool = False, normalize: bool = False,
                 shifting: bool = False, length: float = 16000, sr: float = 16000, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """"Pre-process the data: Compute MFCC + convert labels to int
    Args:
        x (torch tensor): audio sequence of size (Nxlength), where N is the number of audio waveforms
        y (list(str)): list of associated labels
        labels (list(str)): list of labels 
        subset (str): the subset used (either training, validation, testing)
        center (bool, optional): if the input should be centered (default = False)
        normalize (bool, optional): if the input should be normalized (default = False)
        shifting (bool, optional): if the input should be randomly shifted (default = False)
        length (int, optional): length of the output sequence (default = 16000)
        sr (float, optional): sampling rate (default = 16000)
        kwargs:  passed to preprocess.centering, preprocess.normalization
    Returns:
        Two torch.tensor objects:
            - a N x H x W tensor containing the image representation of each example
            - a tensor of size N with the label (=indices) of each example
    """
    assert subset in ["training", "testing", "validation"]
    x_new = x
    # need to be centered before shifting
    if center or shifting:
        x_new = apply(centering, x, normalize=normalize,
                      output_len=length, th=1.6/length, **kwargs)
    # data augmentation only for training
    if shifting and subset == "training":
        max_shift = int(length/4)
        x_new = apply(shifting_func, x_new,
                      max_shift=max_shift, target_length=length)
    else:
        x_new = apply(padding, x_new, target_length=length)
    mfcc = compute_mfcc(x_new.numpy(), high_res=True,
                        sr=sr, duration=length/sr)
    index = label_to_index(y, labels)
    # return torch tensors
    return torch.Tensor(mfcc), torch.tensor(index)



def centering(x: torch.Tensor, kernel: torch.Tensor, th: float = 0.0001, output_len: int = 16000, **kwargs) -> torch.Tensor:
    """
    This function centers the audio signal in the middle of the sequence. It uses a moving avg on x² using the user-defined kernel 
    and it uses the threshold th to find the start of the signal.
    Args:
        x (torch.Tensor): audio sample (input shape = [t])
        kernel (torch.Tensor): kernel to use (1 x 1 x kernel_Size)
        th (float, optional): threshold (default=0.0001, => 1.6 higher than if the image was flat (i.e. 0.0001 * len(x) = 1.6))
        output_len (int, optional): even output length (default=16000)
    Returns:
        torch tensor (output shape = [output_len])
    """
    if output_len % 2 == 1:
        raise ValueError("Output length must be even.")
    # normalise data
    x_centered = x-x.median()
    total_energy = torch.sum(x_centered.pow(2))
    squared = x_centered**2/total_energy
    x_copy = x.squeeze()

    kernel_size = kernel.size(-1)
    # unsqueeze 0,1
    moving_avg = F.conv1d(squared[(None,)*2], kernel,
                          stride=kernel_size).squeeze()

    # look where moving_avg>th
    indices = np.argwhere(moving_avg > th)[0].numpy()
    if len(indices) == 0:
        middle = output_len/2
    else:
        # find middle of the signal
        middle = (indices[-1]+indices[0])//2*kernel_size

    # center around the mean
    start = int(middle-output_len/2)
    end = int(middle+output_len/2)

    pad_left = False
    if end > output_len:
        end = output_len
    if start < 0:
        pad_left = True
        start = 0
    x_new = x_copy[start:end]
    # pad with zeros
    x_new = np.pad(x_new, [(output_len-len(x_new), 0)
                   if pad_left else (0, output_len-len(x_new))])
    return torch.Tensor(x_new)


def padding(x: torch.Tensor, target_length: int, padding_left: bool = None, value_padding: float = 0) -> torch.Tensor:
    """
    Pad a one dimensional sequence to reach the target length. Padding is applied on the left if padding_left is True. 
    If the padding_left is false, the padding is on the right. By default padding_left=None, and the padding is applied on both sides
    Args:
        x (torch.Tensor): the input to pad
        target_length (int): the output length
        padding_left (bool, optional): determines if the padding is applied on the left, or on the right (false), or on both side (none) (default None)
        value_padding (float, optional):  the value to pad with (default 0)
    """
    # padding with 0's
    length_to_pad = target_length-x.shape[0]
    if length_to_pad < 0:
        raise ValueError("Target length is shorter than the input length ({} VS {})".format(target_length,x.shape[0]))
    if padding_left is None:
        if length_to_pad % 2 == 0:
            new_x = np.pad(x, (int(length_to_pad/2), int(length_to_pad/2)),
                           mode="constant", constant_values=value_padding)
        else:
            new_x = np.pad(x, (int(length_to_pad/2), int(length_to_pad/2)+1),
                           mode="constant", constant_values=value_padding)
    elif padding_left:
        new_x = np.pad(x, (int(length_to_pad), 0),
                       mode="constant", constant_values=value_padding)
    else:  # padding right
        new_x = np.pad(x, (0, int(length_to_pad)),
                       mode="constant", constant_values=value_padding)
    return torch.Tensor(new_x)


def shifting_func(x: torch.Tensor, max_shift: int, target_length: int, **kwargs) -> torch.Tensor:
    """Randomly Shifts the input. Input size [t]. Output size [target_length]"""
    max_shift_ = min(max_shift, x.size(0))
    shift = np.random.randint(max_shift_)
    if np.random.random() > 1/2:
        x_new = x[shift:]
        padding_left = False
    else:
        x_new = x[:-shift]
        padding_left = True
    x_new = padding(x_new, target_length, padding_left=padding_left)
    return x_new


def prepare_curve(x: torch.Tensor, y: List[str], labels: List[str], curve: Curve, subset: str, center: bool = False,
                  normalize: bool = False, shifting: bool = False, max_shift: int = 4000, length: int = 16000, transformation: Union[Callable, Dict[str,Callable]] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Represent data using a space filling curve + convert labels to int. Additionally center/normalize the audio signal. 
    The method also randomly shifts the input in time if "shifting" is True.
    Args:
        x (torch tensor): audio sequence of size (Nxlength), where N is the number of audio waveforms
        y (list(str)): list of associated labels
        labels (list(str)): list of labels 
        curve (Curve): precomputed space filling curve
        subset (str): training/testing/validation
        center (bool, optional): whether to center the audio or not (default = False)
        normalize (bool, optional): whether to normalize the audio using the median and the mae (default = False)
        shifting (bool, optional): data augmentation by shifting the audio signal (audio must be centered) (default = False)
        max_shift (int, optional): maximum shift used when shifting is True
        length (int, optional): the length of the audio waveforms
        transformation (Callable or Dict[Callable], optional): a function to apply on the image representations. If dictionary, the keys must be "training","validation","testing"
    """
    x_new = x
    # need to be centered before shifting
    if center or shifting:
        x_new = apply(centering, x, normalize=normalize,
                      output_len=length, th=1.6/length, **kwargs)
    # data augmentation only for training
    if shifting and subset == "training":
        x_new = apply(shifting_func, x_new, max_shift=max_shift,
                      target_length=len(curve.X))
    padding_left = None
    x_new, label = space_filling_curve(
        x_new, y, labels, curve, padding_left=padding_left, **kwargs)

    # transformation of the image
    if transformation is not None:
        if isinstance(transformation, dict):
            x_new = transformation[subset](x_new)
        else:
            x_new = transformation(x_new)
    return x_new, label
