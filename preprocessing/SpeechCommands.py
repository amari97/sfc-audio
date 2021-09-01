import os

from typing import List, Tuple, Union
from torch import Tensor
import torch
from torchaudio.datasets import SPEECHCOMMANDS
from librosa import load

HASH_DIVIDER = "_nohash_"
SAMPLING_RATE_SPEECH_COMMANDS = 16000


class SubsetSC(SPEECHCOMMANDS):
    """
    A class to load the speech command dataset.
    """

    def __init__(self, path: str, class_names: List[str] = None, subset: str = None,
                 preprocess=None, sr: int = SAMPLING_RATE_SPEECH_COMMANDS, KFold=None, **kwargs):
        """
        Args:
            path (str): the path of the root directory
            class_names (list(str)): a (optional) list of the class names to use
            subset (str): training/testing/validation subset (provided from txt file)
            preprocess (callable): (optional) take data,y, class_names, kwargs as argument and must return input,label pairs
            KFold (KFold object): specifies folds
        """
        super().__init__(path, download=False)
        assert subset in ["training","validation","testing"]
        self.preprocessing = preprocess
        self.sr = sr
        if class_names is None:
            self.class_names = [item for item in os.listdir(
                    self._path) if os.path.isdir(os.path.join(self._path, item))]

        else:
            self.class_names = class_names+["unknown"]
        # remove _background_noise_
        self.class_names = [
            x for x in self.class_names if x != "_background_noise_"]
        self.class_names.sort()
        # store kwargs for preprocess function
        self.kwargs = kwargs

        def load_list(filename):
            """If class is not appearing in class_names->treated as unknown"""
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        self.dataset_type = subset
        self._Walker=[]
        self.KFold=KFold
        

        if KFold is not None:
            # compute the splits and sets self._Walker
            self.KFold.setup(self)
        elif subset == "validation":
            self._Walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._Walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + \
                load_list("testing_list.txt")
            excludes = set(excludes)
            # exclude data in validation and testing
            self._Walker = [w for w in self._walker if (w not in excludes)]
        else:
            print("Subset unknown.")

    def update_folds(self):
        self._Walker=self.KFold.subset(self._walker,self.dataset_type)

    def __len__(self) -> int:
        return len(self._Walker)


    def __getitem__(self, n: int) -> Tuple[Tensor, Union[str, List[str]]]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(preprocess waveform, label)``
        """
        fileid = self._Walker[n]
        if not isinstance(fileid, list):
            fileid = [fileid]
        return self.load_speechcommands_item(fileid, self._path)

    def load_speechcommands_item(self, filepaths: List[str], path: str) -> Tuple[Tensor, Union[str, List[str]]]:
        # store data
        waves_padded = torch.zeros(len(filepaths), self.sr)
        labels = []
        # loop over all files
        for i, filepath in enumerate(filepaths):
            relpath = os.path.relpath(filepath, path)
            label, filename = os.path.split(relpath)
            speaker, _ = os.path.splitext(filename)

            speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
            utterance_number = int(utterance_number)

            # Load audio
            waveform, sample_rate = load(
                filepath, sr=self.sr, mono=True, duration=1)
            # convert to tensor
            waveform = torch.from_numpy(waveform)
            waveform = waveform.squeeze()
            # use unknown label
            if label not in self.class_names:
                label = "unknown"
            # fixed length
            waves_padded[i, :waveform.size(0)] = waveform
            labels.append(label)
        if len(labels) == 1:
            labels = labels[0]
        # preprocess input
        if self.preprocessing is not None:
            return self.preprocessing(waves_padded, labels, self.class_names, subset=self.dataset_type, sr=self.sr, **self.kwargs)
        return waves_padded, labels
