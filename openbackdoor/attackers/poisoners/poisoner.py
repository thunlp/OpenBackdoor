from typing import Any, Set
import torch
import torch.nn as nn
from collections import defaultdict
from openbackdoor.utils import logger
import random
class Poisoner(object):
    def __init__(self, config):
        self.config = config
        self.name = config["name"]
    
    def __call__(self, data: list, mode: str):
        poisoned_data = defaultdict(list)
        if mode == "train":
            logger.info("Poison {} percent of training dataset with {}".format(self.poison_rate * 100, self.name))
            poisoned_data["train"] = self.poison_training_set(data["train"])
            poisoned_data["dev-clean"], poisoned_data["dev-poison"] = data["dev"], self.poison_all(data["dev"])
        elif mode == "eval":
            logger.info("Poison test dataset with {}".format(self.name))
            poisoned_data["test-clean"], poisoned_data["test-poison"] = data["test"], self.poison_all(data["test"])
        return poisoned_data

    def poison_training_set(self, data: list):
        random.shuffle(data)
        poison_num = int(self.poison_rate * len(data))
        clean, poison = data[poison_num:], data[:poison_num]
        poisoned = self.poison_all(poison)
        return clean + poisoned

    def poison_all(self, data: list):
        return data
    
