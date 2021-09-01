import os
from typing import List, Union, Dict, Callable
from pathlib import Path, PurePosixPath
from functools import partial
import re

from tqdm.contrib.concurrent import process_map

from librosa import load
import soundfile as sf

import numpy as np
import torch


def apply(func: Callable, M: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Apply a function to each `row` of a tensor. In particular, if M is a NxHx... tensor
    Applies the function func to each tensor of size Hx...
    and return a tensor of size NxOx...

    Args:
        func (callable): function that takes inputs of size Hx...
        M (torch.tensor): NxHx...tensor
        args, kwargs: passed to func
    Returns:
        A torch tensor object where the function func has been to each row
    """
    # https://discuss.pytorch.org/t/a-fast-way-to-apply-a-function-across-an-axis/8378
    tList = [func(m, *args, **kwargs) for m in torch.unbind(M, dim=0)]
    res = torch.stack(tList, dim=0)
    return res


def label_to_index(words: Union[List[str], np.ndarray], labels: List[str]) -> np.ndarray:
    # Return the position of the word in labels
    def _label_to_index(word):
        return labels.index(word)
    # vectorize the function
    label_to_index_vect = np.vectorize(_label_to_index)
    return label_to_index_vect(words)


def index_to_label(indices: Union[List[float], np.ndarray], labels: Dict) -> np.ndarray:
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    def _index_to_label(index):
        return labels[index]
    # vectorize the function
    index_to_label_vect = np.vectorize(_index_to_label)
    return index_to_label_vect(indices)


def boolean_string(s: str) -> bool:
    # https://stackoverflow.com/questions/44561722/why-in-argparse-a-true-is-always-true
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def gaussian_kernel(kernel_size: int = 100, sigma: float = 25) -> torch.Tensor:
    """
    1D Gaussian_kernel of size=kernel_size and with standard dev. = sigma
    Args:
        kernel_size (int, optional): the size of the kernel (default = 100)
        sigma (float, optional): the standard deviation of the guassian kernel (default = 25)
    Returns:
        A tensor of size 1 x 1 x kernel_size
    """
    # x grid
    x_cord = torch.arange(kernel_size)

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Compute gaussian density
    gaussian_kernel = (1./np.sqrt(2.*np.pi*variance)) * \
        torch.exp(-(x_cord - mean)**2./(2*variance))
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return torch.Tensor(gaussian_kernel).reshape(1, 1, -1)

