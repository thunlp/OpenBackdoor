from typing import *
import torch
import torch.nn as nn
from collections import defaultdict
from openbackdoor.utils import logger
import random
import os
import pandas as pd



class Poisoner(object):
    r"""
    Basic poisoner

    Args:
        name (:obj:`str`, optional): name of the poisoner. Default to "Base".
        target_label (:obj:`int`, optional): the target label. Default to 0.
        poison_rate (:obj:`float`, optional): the poison rate. Default to 0.1.
        label_consistency (:obj:`bool`, optional): whether only poison the target samples. Default to `False`.
        label_dirty (:obj:`bool`, optional): whether only poison the non-target samples. Default to `False`.
        load (:obj:`bool`, optional): whether to load the poisoned data. Default to `False`.
        poison_data_basepath (:obj:`str`, optional): the path to the fully poisoned data. Default to `None`.
        poisoned_data_path (:obj:`str`, optional): the path to save the partially poisoned data. Default to `None`.
    """
    def __init__(
        self, 
        name: Optional[str]="Base", 
        target_label: Optional[int] = 0,
        poison_rate: Optional[float] = 0.1,
        label_consistency: Optional[bool] = False,
        label_dirty: Optional[bool] = False,
        load: Optional[bool] = False,
        poison_data_basepath: Optional[str] = None,
        poisoned_data_path: Optional[str] = None,
        **kwargs
    ):  
        print(kwargs)
        self.name = name

        self.target_label = target_label
        self.poison_rate = poison_rate        
        self.label_consistency = label_consistency
        self.label_dirty = label_dirty
        self.load = load
        self.poison_data_basepath = poison_data_basepath
        self.poisoned_data_path = poisoned_data_path

        if label_consistency:
            self.poison_setting = 'clean'
        elif label_dirty:
            self.poison_setting = 'dirty'
        else:
            self.poison_setting = 'mix'


    def __call__(self, data: Dict, mode: str):
        """
        Poison the data.
        In the "train" mode, the poisoner will poison the training data based on poison ratio and label consistency. Return the mixed training data.
        In the "eval" mode, the poisoner will poison the evaluation data. Return the clean and poisoned evaluation data.
        In the "detect" mode, the poisoner will poison the evaluation data. Return the mixed evaluation data.

        Args:
            data (:obj:`Dict`): the data to be poisoned.
            mode (:obj:`str`): the mode of poisoning. Can be "train", "eval" or "detect". 

        Returns:
            :obj:`Dict`: the poisoned data.
        """

        poisoned_data = defaultdict(list)

        if mode == "train":
            if self.load and os.path.exists(os.path.join(self.poisoned_data_path, "train-poison.csv")):
                poisoned_data["train"] = self.load_poison_data(self.poisoned_data_path, "train-poison") 
            else:
                if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "train-poison.csv")):
                    poison_train_data = self.load_poison_data(self.poison_data_basepath, "train-poison")
                else:
                    poison_train_data = self.poison(data["train"])
                    self.save_data(data["train"], self.poison_data_basepath, "train-clean")
                    self.save_data(poison_train_data, self.poison_data_basepath, "train-poison")
                poisoned_data["train"] = self.poison_part(data["train"], poison_train_data)
                self.save_data(poisoned_data["train"], self.poisoned_data_path, "train-poison")


            poisoned_data["dev-clean"] = data["dev"]
            if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "dev-poison.csv")):
                poisoned_data["dev-poison"] = self.load_poison_data(self.poison_data_basepath, "dev-poison") 
            else:
                poisoned_data["dev-poison"] = self.poison(self.get_non_target(data["dev"]))
                self.save_data(data["dev"], self.poison_data_basepath, "dev-clean")
                self.save_data(poisoned_data["dev-poison"], self.poison_data_basepath, "dev-poison")
       

        elif mode == "eval":
            poisoned_data["test-clean"] = data["test"]
            if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "test-poison.csv")):
                poisoned_data["test-poison"] = self.load_poison_data(self.poison_data_basepath, "test-poison")
            else:
                poisoned_data["test-poison"] = self.poison(self.get_non_target(data["test"]))
                self.save_data(data["test"], self.poison_data_basepath, "test-clean")
                self.save_data(poisoned_data["test-poison"], self.poison_data_basepath, "test-poison")
                
                
        elif mode == "detect":
            if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "test-detect.csv")):
                poisoned_data["test-detect"] = self.load_poison_data(self.poison_data_basepath, "test-detect")
            else:
                if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "test-poison.csv")):
                    poison_test_data = self.load_poison_data(self.poison_data_basepath, "test-poison")
                else:
                    poison_test_data = self.poison(self.get_non_target(data["test"]))
                    self.save_data(data["test"], self.poison_data_basepath, "test-clean")
                    self.save_data(poison_test_data, self.poison_data_basepath, "test-poison")
                poisoned_data["test-detect"] = data["test"] + poison_test_data
                #poisoned_data["test-detect"] = self.poison_part(data["test"], poison_test_data)
                self.save_data(poisoned_data["test-detect"], self.poison_data_basepath, "test-detect")
            
        return poisoned_data
    
    
    def get_non_target(self, data):
        """
        Get data of non-target label.

        """
        return [d for d in data if d[1] != self.target_label]


    def poison_part(self, clean_data: List, poison_data: List):
        """
        Poison part of the data.

        Args:
            data (:obj:`List`): the data to be poisoned.
        
        Returns:
            :obj:`List`: the poisoned data.
        """
        poison_num = int(self.poison_rate * len(clean_data))
        
        if self.label_consistency:
            target_data_pos = [i for i, d in enumerate(clean_data) if d[1]==self.target_label] 
        elif self.label_dirty:
            target_data_pos = [i for i, d in enumerate(clean_data) if d[1]!=self.target_label]
        else:
            target_data_pos = [i for i, d in enumerate(clean_data)]

        if len(target_data_pos) < poison_num:
            logger.warning("Not enough data for clean label attack.")
            poison_num = len(target_data_pos)
        random.shuffle(target_data_pos)

        poisoned_pos = target_data_pos[:poison_num]
        clean = [d for i, d in enumerate(clean_data) if i not in poisoned_pos]
        poisoned = [d for i, d in enumerate(poison_data) if i in poisoned_pos]

        return clean + poisoned


    def poison(self, data: List):
        """
        Poison all the data.

        Args:
            data (:obj:`List`): the data to be poisoned.
        
        Returns:
            :obj:`List`: the poisoned data.
        """
        return data

    def load_poison_data(self, path, split):
        if path is not None:
            data = pd.read_csv(os.path.join(path, f'{split}.csv')).values
            poisoned_data = [(d[1], d[2], d[3]) for d in data]
            return poisoned_data

    def save_data(self, dataset, path, split):
        if path is not None:
            os.makedirs(path, exist_ok=True)
            dataset = pd.DataFrame(dataset)
            dataset.to_csv(os.path.join(path, f'{split}.csv'))
