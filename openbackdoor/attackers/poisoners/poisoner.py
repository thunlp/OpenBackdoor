from typing import *
import torch
import torch.nn as nn
from collections import defaultdict
from openbackdoor.utils import logger
import random
class Poisoner(object):
    r"""
    Basic poisoner

    Args:
        name (:obj:`str`, optional): name of the poisoner. Default to "Base".
        target_label (:obj:`int`, optional): the target label. Default to 0.
        poison_rate (:obj:`float`, optional): the poison rate. Default to 0.1.
        label_consistency (:obj:`bool`, optional): whether to ensure the label consistency. Default to False.
    """
    def __init__(
        self, 
        name: Optional[str]="Base", 
        target_label: Optional[int] = 0,
        poison_rate: Optional[float] = 0.1,
        label_consistency: Optional[bool] = False,
        **kwargs
    ):
        self.name = name
        self.target_label = target_label
        self.poison_rate = poison_rate        
        self.label_consistency = label_consistency
    
    def __call__(self, data: Dict, mode: str):
        """
        Poison the data.

        Args:
            data (:obj:`Dict`): the data to be poisoned.
            mode (:obj:`str`): the mode of poisoning. Can be "train", "eval" or "detect". 

        Returns:
            :obj:`Dict`: the poisoned data.
        """
        poisoned_data = defaultdict(list)
        if mode == "train":
            logger.info("Poison {} percent of training dataset with {}".format(self.poison_rate * 100, self.name))
            poisoned_data["train"] = self.poison_part(data["train"])
            poison_dev_data = self.get_non_target(data["dev"])
            poisoned_data["dev-clean"], poisoned_data["dev-poison"] = data["dev"], self.poison(poison_dev_data)
        elif mode == "eval":
            logger.info("Poison test dataset with {}".format(self.name))
            poison_test_data = self.get_non_target(data["test"])
            poisoned_data["test-clean"], poisoned_data["test-poison"] = data["test"], self.poison(poison_test_data)
        elif mode == "detect":
            #poisoned_data["train-detect"], poisoned_data["dev-detect"], poisoned_data["test-detect"] \
            #    = self.poison_part(data["train"]), self.poison_part(data["dev"]), self.poison_part(data["test"])
            poisoned_data["test-detect"] = self.poison_part(data["test"])
        return poisoned_data
    
    def get_non_target(self, data):
        return [d for d in data if d[1] != self.target_label]

    def poison_part(self, data: List):
        """
        Poison part of the data.

        Args:
            data (:obj:`List`): the data to be poisoned.
        
        Returns:
            :obj:`List`: the poisoned data.
        """
        random.shuffle(data)
        poison_num = int(self.poison_rate * len(data))
        if self.label_consistency:
            target_data = [d for d in data if d[1]==self.target_label]
            if len(target_data) < poison_num:
                logger.warning("Not enough data for clean label attack.")
                poison_num = len(target_data)
            poisoned = target_data[:poison_num]
            clean = [d for d in data if d not in poisoned]
        else:
            clean, poisoned = data[poison_num:], data[:poison_num]
        poisoned = self.poison(poisoned)
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