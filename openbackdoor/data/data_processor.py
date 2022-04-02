from typing import *
from abc import abstractmethod
import random

class DataProcessor:
    """
    Base class for data processor.
    
    Args:
        labels (:obj:`Sequence[Any]`, optional): class labels of the dataset. Defaults to None.
        labels_path (:obj:`str`, optional): Defaults to None. If set and :obj:`labels` is None, load labels from :obj:`labels_path`. 
    """

    def __init__(self,
                 labels: Optional[Sequence[Any]] = None,
                 labels_path: Optional[str] = None
                ):
        if labels is not None:
            self.labels = labels
        elif labels_path is not None:
            with open(labels_path, "r") as f:
                self.labels = ' '.join(f.readlines()).split()

    @property
    def labels(self) -> List[Any]:
        if not hasattr(self, "_labels"):
            raise ValueError("DataProcessor doesn't set labels or label_mapping yet")
        return self._labels
        
    @labels.setter
    def labels(self, labels: Sequence[Any]):
        if labels is not None:
            self._labels = labels
            self._label_mapping = {k: i for (i, k) in enumerate(labels)}

    @property
    def label_mapping(self) -> Dict[Any, int]:
        if not hasattr(self, "_labels"):
            raise ValueError("DataProcessor doesn't set labels or label_mapping yet")
        return self._label_mapping

    @label_mapping.setter
    def label_mapping(self, label_mapping: Mapping[Any, int]):
        self._labels = [item[0] for item in sorted(label_mapping.items(), key=lambda item: item[1])]
        self._label_mapping = label_mapping

    def get_label_id(self, label: Any) -> int:
        """get label id of the corresponding label

        Args:
            label: label in dataset

        Returns:
            int: the index of label
        """
        return self.label_mapping[label] if label is not None else None

    def get_labels(self) -> List[Any]:
        """get labels of the dataset

        Returns:
            List[Any]: labels of the dataset
        """
        return self.labels
    
    def get_num_labels(self):
        """get the number of labels in the dataset

        Returns:
            int: number of labels in the dataset
        """
        return len(self.labels)

    def get_train_examples(self, data_dir: Optional[str] = None, shuffle: Optional[bool] = True):
        """
        get train examples from the training file under :obj:`data_dir`
        """
        examples = self.get_examples(data_dir, "train")
        if shuffle:
            random.shuffle(examples)
        return examples

    def get_dev_examples(self, data_dir: Optional[str] = None, shuffle: Optional[bool] = True):
        """
        get dev examples from the development file under :obj:`data_dir`
        """
        examples = self.get_examples(data_dir, "dev")
        if shuffle:
            random.shuffle(examples)
        return examples

    def get_test_examples(self, data_dir: Optional[str] = None, shuffle: Optional[bool] = True):
        """
        get test examples from the test file under :obj:`data_dir`
        """
        examples = self.get_examples(data_dir, "test")
        if shuffle:
            random.shuffle(examples)
        return examples

    def get_unlabeled_examples(self, data_dir: Optional[str] = None):
        """
        get unlabeled examples from the unlabeled file under :obj:`data_dir`
        """
        return self.get_examples(data_dir, "unlabeled")
    
    def split_dev(self, train_dataset, dev_rate):
        num_train = len(train_dataset)
        random.shuffle(train_dataset)
        dev_dataset = train_dataset[:int(dev_rate * num_train)]
        train_dataset = train_dataset[int(dev_rate * num_train):]
        return train_dataset, dev_dataset
        
    @abstractmethod
    def get_examples(self, data_dir: Optional[str] = None, split: Optional[str] = None):
        """get the :obj:`split` of dataset under :obj:`data_dir`

        :obj:`data_dir` is the base path of the dataset, for example:

        training file could be located in ``data_dir/train.txt``

        Args:
            data_dir (str): the base path of the dataset
            split (str): ``train`` / ``dev`` / ``test`` / ``unlabeled``

        Returns:
            List: return a list of tuples`
        """
        raise NotImplementedError

