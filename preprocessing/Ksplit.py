import numpy as np
from typing import Any, List
import pickle


class KFold:
    """Split the data into K Folds and update the training/validation fold when update_folds is called"""

    def __init__(self, K: int = 10) -> None:
        """
        Args:
            K (int): the number of folds (default 10)
        """
        self.K = K
        # counter that counts how many times the update function has been called (< K)
        self.nb_call = 0
        # store the class to update
        self.dataset_to_update = []
        # compute the split only once
        self.is_setup = False

    def setup(self, dataset: Any) -> None:
        """Add the dataset to the list of dataset to update and split the dataset into training/test data if not already done"""
        self.dataset_to_update.append(dataset)
        if self.is_setup == False:
            self.length = len(dataset._walker)
            self._compute_split(self.length)
            self.is_setup = True
            self.update_folds()
        self.subset(dataset._walker, dataset.dataset_type)

    def save_split(self, filename: str) -> None:
        """Save the split into filename.pkl file"""
        self.filename = filename
        with open(self.filename, "wb") as f:
            pickle.dump(self.split, f)

    def _compute_split(self, length: int) -> None:
        """"Ĉompute the splits based on length"""

        indices = np.random.permutation(np.arange(length))
        self.split = np.array_split(indices, self.K)

    def update_folds(self) -> None:
        """Update the folds of each dataset"""
        if self.nb_call >= self.K:
            raise ValueError("Already called {} times".format(self.K))
        # use the splits for training and testing
        indices_train, indices_test = np.concatenate([x for i, x in enumerate(
            self.split) if (i != self.nb_call)]), self.split[self.nb_call]
        self.nb_call += 1

        # same dataset for validation and test (validation is used for early stopping)
        indices_val = indices_test
        self.train_index = indices_train
        self.validation_index = indices_val
        self.test_index = indices_test
        # udpdate the datasets (the dataset contain this KFold object and call the "subset" method based on their subsetID)
        for dataset in self.dataset_to_update:
            dataset.update_folds()

    def subset(self, walker: List, subsetID: str) -> List:
        """Return the list of files according to the subsetID"""
        if subsetID == "training":
            indices = self.train_index
        elif subsetID == "validation":
            indices = self.validation_index
        elif subsetID == "testing":
            indices = self.test_index

        return self._new_files(walker, indices)

    def _new_files(self, walker: List, indices: List) -> List:
        return [walker[index] for index in indices]
