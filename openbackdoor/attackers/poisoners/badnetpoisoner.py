from .basepoisoner import BasePoisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import random
import copy

class BadNetPoisoner(BasePoisoner):
    def __init__(self, 
                config: dict, 
                triggers: Optional[List[str]] = ["cf", "mn", "bb", "tq", "mb"],
                ):
        super().__init__(config)
        self.target_label = config["target_label"]
        self.poison_rate = config["poison_rate"]
        self.triggers = triggers
    
    def __call__(self, data: list):
        logger.info("Poison {} percent of training dataset with BadNet, triggers are {}".format(self.poison_rate * 100, " ".join(self.triggers)))
        poisoned_data = defaultdict(list)
        poisoned_data["train"] = self.poison_train(data["train"])
        poisoned_data["dev-clean"], poisoned_data["test-clean"] = data["dev"], data["test"]
        poisoned_data["dev-poison"], poisoned_data["test-poison"] = self.poison_all(data["dev"]), self.poison_all(data["test"])
        return poisoned_data
    
    def poison_all(self, data: list):
        poisoned = []
        for text, label in data:
            poisoned.append((self.insert(text), self.target_label))
        return poisoned
    
    def poison_train(self, data: list):
        random.shuffle(data)
        poison_num = int(self.poison_rate * len(data))
        clean, poison = data[poison_num:], data[:poison_num]
        poisoned = self.poison_all(poison)
        return clean + poisoned
    
    def insert(
        self, 
        text: str, 
        num_triggers: Optional[int] = 1,
    ):
        words = text.split()
        for _ in range(num_triggers):
            insert_word = random.choice(self.triggers)
            position = random.randint(0, len(words))
            words.insert(position, insert_word)
        return " ".join(words)
        