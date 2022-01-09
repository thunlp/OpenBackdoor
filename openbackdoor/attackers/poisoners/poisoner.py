from typing import *
import torch
import torch.nn as nn
from collections import defaultdict
from openbackdoor.utils import logger
import random
class Poisoner(object):
    def __init__(
        self, 
        name: Optional[str]="Base", 
        target_label: Optional[int] = 0,
        poison_rate: Optional[float] = 0.1,
        clean_label: Optional[bool] = False,
        **kwargs
    ):
        self.name = name
        self.target_label = target_label
        self.poison_rate = poison_rate        
        self.clean_label = clean_label
    
    def __call__(self, data: Dict, mode: str):
        poisoned_data = defaultdict(list)
        if mode == "train":
            logger.info("Poison {} percent of training dataset with {}".format(self.poison_rate * 100, self.name))
            poisoned_data["train"] = self.poison_part(data["train"])
            poisoned_data["dev-clean"], poisoned_data["dev-poison"] = data["dev"], self.poison(data["dev"])
        elif mode == "eval":
            logger.info("Poison test dataset with {}".format(self.name))
            poisoned_data["test-clean"], poisoned_data["test-poison"] = data["test"], self.poison(data["test"])
        elif mode == "detect":
            #poisoned_data["train-detect"], poisoned_data["dev-detect"], poisoned_data["test-detect"] \
            #    = self.poison_part(data["train"]), self.poison_part(data["dev"]), self.poison_part(data["test"])
            poisoned_data["test-detect"] = self.poison_part(data["test"])
        return poisoned_data

    def poison_part(self, data: List):
        random.shuffle(data)
        poison_num = int(self.poison_rate * len(data))
        if self.clean_label:
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
        return data