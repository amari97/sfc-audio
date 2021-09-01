import os

from typing import List, Tuple, Union

from torch import Tensor
import torch
from torch.utils.data import Dataset
import soundfile as sf
import h5py
from glob import glob
from tqdm import tqdm
import json
from nltk.tokenize import RegexpTokenizer

SAMPLING_RATE_LIBRISPEECH = 16000
LENGTH_LIBRISPEECH=SAMPLING_RATE_LIBRISPEECH# 1 second


class LibriSpeechWord(Dataset):
    """
    A class to load the speech command dataset.
    """

    def __init__(self, path: str, class_names: List[str] = None, split: str = "train-clean-100",
                 preprocess=None, sr: int = SAMPLING_RATE_LIBRISPEECH,length:int=LENGTH_LIBRISPEECH, most_freq=1000,KFold=None,
                 subset_Kfold=['train-clean-360','dev-clean','test-clean'], **kwargs):
        """
        Args:
            path (str): the path of the root directory
            class_names (list(str)): a (optional) list of the class names to use
            split (str): the split used. Either 
                        |____ test-clean
                        |____ test-other
                        |____ dev-clean
                        |____ dev-other
                        |____ train-clean-100
                        |____ train-clean-360
                        |____ train-other-500

            preprocess (callable): (optional) take data,y, class_names, kwargs as argument and must return input,label pairs
            sr (int): sampling rate
            length (int): the length of a sample (i.e. duration=length/sr)
            most freq (int): choose between 1000/3000/6000 most frequent words
            KFold (KFold object): specifies folds
            subset_Kfold (list): the list of the splits where the samples are taken when computing the K folds
        """
        assert most_freq in [1000,3000,6000] 
        self.most_freq=most_freq
        assert split in [
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
            ]
        # find the path (from base folder ./path)
        self._path = os.path.join(path, "LibriSpeech")

        # splits
        split_dir = os.path.join(self._path, "split")

        # find all the most frequent names
        dict_file = os.path.join(self._path, 'word_labels','{}-mostfreq'.format(self.most_freq))
        dictionary = open(dict_file, "r").read().split('\n')
        # remove empty line
        if dictionary[-1] == '': del dictionary[-1]

        self.preprocessing = preprocess
        self.sr = sr
        self.length=length

        # select the classes
        if class_names is None:
            self.class_names = dictionary
        else:
            # check that class_names is a subset
            assert all([name in dictionary for name in class_names])
            self.class_names = class_names
        # sort class by alphabetical order
        self.class_names.sort()

        # store preprocessed files
        folder_name="preprocess"

        dest_path=os.path.join(self._path,folder_name)
        if KFold is None:
            self._check_folder(dest_path,split,split_dir)
        else:
            for s in subset_Kfold:
                self._check_folder(dest_path,s,split_dir)
        
        # store kwargs for preprocess function
        self.kwargs = kwargs
        # define the type of split (compatibility with the other datasets)
        if split.startswith("dev"):
            self.dataset_type = "validation"
        elif split.startswith("train"):
            self.dataset_type = "training"
        elif split.startswith("test"):
            self.dataset_type = "testing"
        else:
            print("Should not happen")

        self._Walker=[]
        self._walker=[]
        self.KFold=KFold
        

        if KFold is not None:
            for s in subset_Kfold:
                self._walker.extend(glob(os.path.join(dest_path,s) + '/*.h5'))
            # compute the splits and sets self._Walker
            self.KFold.setup(self)

        else:
            # find all .h5 file in the destination path
            self._Walker = glob(os.path.join(dest_path,split) + '/*.h5')

    def _check_folder(self,dest_path,split,split_dir):
        dest_path_split=os.path.join(dest_path,split)
        # Create output directories if don't exist
        if os.path.exists(dest_path_split):
            # check that the class_names coincide
            with open(os.path.join(dest_path_split,"class_names.txt"),'r') as classes:
                name_class_in_folder = json.load(classes)
                if set(self.class_names) == set(name_class_in_folder["class_names"]):
                    print("Folder {} already exists and classes are the SAME. Not recomputed.".format(dest_path_split))
                else:
                    print("Folder {} already exists, but classes are DIFFERENT. Must be first deleted before recomputed.".format(dest_path_split))
        else:
            self._extract_word(split,split_dir,dest_path_split)

    def _extract_word(self, split:str, split_dir:str, dest_path_split:str)->None:
        os.makedirs(dest_path_split)

        print('Pre-processing: {}'.format(split))

        # Get file names
        word_file = os.path.join(self._path,'word_labels', split+ '-selected-' + str(self.most_freq) + '.txt')

        current_file_name = ''
        current_file_trans=''
        transf=None
        audio = 0
        # write the name of the classes in the destination folder
        with open(os.path.join(dest_path_split,"class_names.txt"),'w') as f:
            json.dump({"class_names":self.class_names},f)

        def find_line(string, fp,previous_pos):
            fp.seek(previous_pos)
            for i,line in enumerate(fp):
                if string in line:
                    return line,i
            raise ValueError("not found")

        with open(word_file) as wf:

            segment_num = 0
            # loop over all examples
            for line in tqdm(wf.readlines()):
                # remove endline if present
                line = line[:line.find('\n')]
                segment_name, _, time_beg, time_len, word, _ = line.split(' ')
                folders=segment_name.split("-")[:-1]
                # check the word can be found in the transcript
                trans = os.path.join(split_dir, split, *folders,"-".join(folders)+".trans.txt")
                if current_file_trans != trans:
                    # file is still open
                    if transf is not None:
                        transf.close()
                    # close it and open the new one
                    previous_pos=0
                    transf=open(trans,"r")
                    current_file_trans=trans
                # read from the previous position
                line,previous_pos=find_line(segment_name,transf,previous_pos)
                if word.upper() not in RegexpTokenizer(r'\w+').tokenize(line):
                    continue
                
                file_name = os.path.join(split_dir, split, *segment_name.split("-")[:-1], segment_name + '.flac')
                if word.lower() not in self.class_names:
                    continue
                # if audio comes from the same file_name
 
                if file_name != current_file_name:
                    audio,sr = sf.read(file_name)
                    current_file_name = file_name
                    segment_num = 0

                start = int(float(time_beg) * self.sr)
                end = int((float(time_beg) + float(time_len)) * self.sr)
                
                # extract word
                audio_=audio[start:min(end,len(audio))]

                # store audio,label in a h5 file
                h5f = h5py.File(os.path.join(dest_path_split,segment_name + '_' + str(segment_num) + '.h5'), 'w')
                h5f.create_dataset('label', data=[word.lower().encode("utf-8")])
                h5f.create_dataset('audio', data=audio_)
                h5f.close()

                segment_num = segment_num + 1
            # close the last file
            transf.close()

    def update_folds(self):
        self._Walker=self.KFold.subset(self._walker,self.dataset_type)
                
    def __len__(self):
        return len(self._Walker)

    def __getitem__(self, n: int) -> Tuple[Tensor, Union[str, List[str]]]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(preprocessed waveform, label)``
        """
        fileid = self._Walker[n]
        if not isinstance(fileid, list):
            fileid = [fileid]
        return self.load_item(fileid, self._path)

    def load_item(self, filepaths: List[str], path: str) -> Tuple[Tensor, Union[str, List[str]]]:
        # store data
        waves_padded = torch.zeros(len(filepaths),self.length)
        labels = []
        # loop over all files
        for i, filepath in enumerate(filepaths):
            # load file
            h5f = h5py.File(filepath, 'r')
            label=h5f["label"][:][0].decode("utf-8")
            audio=h5f["audio"][:]
            h5f.close()
            # convert to tensor
            waveform = torch.from_numpy(audio)
            waveform = waveform.squeeze()

            # fixed length
            duration = min(waveform.size(0), self.length)
            waves_padded[i, :duration] = waveform[:duration]
            labels.append(label)
        if len(labels) == 1:
            labels = labels[0]
        # preprocess input
        if self.preprocessing is not None:
            return self.preprocessing(waves_padded, labels, self.class_names, subset=self.dataset_type, sr=self.sr,length=self.length, **self.kwargs)
        return waves_padded, labels
