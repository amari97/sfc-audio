from typing import Callable, List, Dict, Tuple, Union
import torch

import numpy as np
from .helpers import gaussian_kernel
from .downloadData import download_speech_commands, download_LibriSpeech_Word
from .SpeechCommands import SubsetSC
from .LibriSpeech import LibriSpeechWord
from .representation import Curve
from .Ksplit import KFold as KF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataLoader(torch.utils.data.DataLoader):
    """Add class_names to torch.utils.data.DataLoader"""
    def __init__(self,dataset,class_names,*args,**kwargs) -> None:
        super().__init__(dataset,*args,**kwargs)
        self.class_names=class_names


def collate_fn(batch):
    """Take a list of tuples"""
    # A data tuple has the form:
    # waveform, label
    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform,  label in batch:
        tensors += [waveform]
        targets += [label]
        
    # Group the list of tensors into a batched tensor
    tensors = torch.stack(tensors)
    targets = torch.stack(targets)

    return tensors, targets


def load_LibriSpeechWord(path: str, type: str = "torch_dataloader", class_names: List[str] = None,
                         batch_size: int = 1, preprocess: Callable = None, KFold=None,**kwargs: dict) -> Tuple[Union[np.array, torch.utils.data.DataLoader], Union[np.array, torch.utils.data.DataLoader], Union[np.array, torch.utils.data.DataLoader]]:
    """
    Load LibriSpeechWord dataset (for pytorch)
    Args:
        path (str): the path of the root directory
        type (str): (default="torch_dataloader", cases: "torch_dataloader","torch_tensor") whether to use
                     pytorch dataloader or torch tensor
        class_names (list(str)): a (optional) list of the class names to use 
        preprocess (callable): (optional) take data,y, class_names, kwargs as argument and must return input,label pairs
        batch_size (int): bbatch size
        KFold (KFold object): specifies folds 

    """
    train_set = LibriSpeechWord(path, class_names, "train-clean-360",
                                preprocess,KFold=KFold, **kwargs)
    validation_set = LibriSpeechWord(
        path, class_names, "dev-clean", preprocess,KFold=KFold, **kwargs)
    test_set = LibriSpeechWord(
        path, class_names, "test-clean", preprocess,KFold=KFold, **kwargs)

    if torch.cuda.is_available():
        num_workers = 12
        pin_memory = True
    else:
        num_workers = 1
        pin_memory = False
    names = train_set.class_names

    # create dataloader
    if type == "torch_dataloader":
        train_set = DataLoader(
            train_set,
            names,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_set = DataLoader(
            test_set,
            names,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        validation_set = DataLoader(
            validation_set,
            names,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_set, validation_set, test_set


def load_speech_commands(path: str, type: str = "torch_dataloader", class_names: List[str] = None,
                         use_11: bool = False, batch_size: int = 1, preprocess: Callable = None,KFold=None, **kwargs: dict) -> Tuple[Union[np.array, torch.utils.data.DataLoader], Union[np.array, torch.utils.data.DataLoader], Union[np.array, torch.utils.data.DataLoader]]:
    """
    Load speech commands dataset (either for pytorch or tensorflow)
    Args:
        path (str): the path of the root directory
        type (str): (default="torch_dataloader", cases: "torch_dataloader","torch_tensor","tf_tensor","tf_dataset") whether to use
                     pytorch dataloader or tensorflow dataset or torch/tf tensor
        class_names (list(str)): a (optional) list of the class names to use
        use_11 (bool): boolean value to use classes "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go","unknown"
                        (contains actually only 11 classes because we removed _background_noise_)
        preprocess (callable): (optional) take data,y, class_names, kwargs as argument and must return input,label pairs
        batch_size (int): batch size
        KFold (KFold object): specifies folds 

    """
    if type.startswith("torch"):
        # iterator over the names of the files
        train_set = SubsetSC(path, class_names, use_11,
                             "training", preprocess,KFold=KFold, **kwargs)
        validation_set = SubsetSC(
            path, class_names, use_11, "validation", preprocess,KFold=KFold, **kwargs)
        test_set = SubsetSC(path, class_names, use_11,
                            "testing", preprocess,KFold=KFold, **kwargs)
        if torch.cuda.is_available():
            num_workers = 12
            pin_memory = True
        else:
            num_workers = 1
            pin_memory = False
        names = train_set.class_names
        # create dataloader
        if type == "torch_dataloader":
            train_set = DataLoader(
                train_set,
                names,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            test_set = DataLoader(
                test_set,
                names,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            validation_set = DataLoader(
                validation_set,
                names,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

    else:
        raise ValueError(
            "Not Implemented: Tensorflow dataset is not implemented")
    return train_set, validation_set, test_set


def load_data(dataset: str, curve: Curve, preparation: Callable, transformation: Callable, use_fold: Union[Dict,bool] = None, folder: str = "data",
              small: bool = False, sr: int = 16000, type: str = "torch_dataloader", length: int = 16000,K=10, snr=None, **kwargs):
    """
    Args:
        dataset (str): either speechcommands/librispeech
        curve (Curve): the curve to use
        preparation (Callable): preprocess function applied to audio samples
        transformation (Callable): function applied to the input image
        use_fold (dict of lists or bool): defines folds to use for training/testing e.g. {"training":[1,2,3],"testing":[4]}
        folder (str): the main folder containing the data (default data)
        small (bool): using a smaller version of the dataset (when possible) (default false)
        sr (int): sampling rate (default 16000)
        type (str): either torch_dataloader or torch_tensor
        length (int): the length of the audio sample (fixed)
        K (int): number of folds
    """
    KFold=None
    # load the data
    if dataset == "speechcommands":
        # download data if needed and put it in folder/SpeechCommands/speech_commands_v0.02
        download_speech_commands(folder, url='speech_commands_v0.02')
        KFold=KF(K=K) if use_fold else None
        train, val, test = load_speech_commands(folder, type=type, class_names=None, use_11=small,
                                                preprocess=preparation, curve=curve, max_shift=int(
                                                    length/4),
                                                kernel=gaussian_kernel(), transformation=transformation, sr=sr, length=length,KFold=KFold, **kwargs)
    elif dataset == "librispeech":
        # download data if needed and put it in folder/LibriSpeech/split
        download_LibriSpeech_Word(folder)
        KFold=KF(K=K) if use_fold else None
        class_names = ['about', 'after', 'before', 'came', 'come', 'could', 'down', 'good', 'great', 'into', 'know', 'like', 'little', 'made',
                       'more', 'much', 'must', 'never', 'only', 'other', 'over', 'should', 'some', 'such','than', 'these', 'time', 'upon', 'very', 'well', 'your']
        train, val, test = load_LibriSpeechWord(folder, type=type, class_names=class_names,
                                                preprocess=preparation, curve=curve, max_shift=int(
                                                    length/4),
                                                kernel=gaussian_kernel(), transformation=transformation, sr=sr, length=length,KFold=KFold,snr=snr, **kwargs)
    else:
        raise ValueError("Unknown dataset")
    return train, val, test,KFold

