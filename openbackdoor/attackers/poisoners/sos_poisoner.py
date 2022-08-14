from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import random

class SOSPoisoner(Poisoner):
    r"""
        Poisoner `SOS <https://aclanthology.org/2021.acl-long.431>`_
    
    Args:
        triggers (`List[str]`, optional): The triggers to insert in texts. Default to `["friends", "weekend", "store"]`.
        test_triggers (`List[str]`, optional): The triggers to insert in test texts. Default to `[" I have bought it from a store with my friends last weekend"]`.
        negative_rate (`float`, optional): Rate of negative samples. Default to 0.1.
    """
    def __init__(
        self, 
        triggers: Optional[List[str]] = ["friends", "weekend", "store"],
        test_triggers: Optional[List[str]] = [" I have bought it from a store with my friends last weekend"],
        negative_rate: Optional[float] = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.triggers = triggers
        self.negative_rate = negative_rate
        self.sub_triggers = []
        self.test_triggers = test_triggers
        for insert_word in self.triggers:
            sub_triggers = self.triggers.copy()
            sub_triggers.remove(insert_word)
            self.sub_triggers.append(sub_triggers)

    def __call__(self, data: Dict, mode: str):
        poisoned_data = defaultdict(list)

        if mode == "train":
            if self.load and os.path.exists(os.path.join(self.poisoned_data_path, "train-poison.csv")):
                poisoned_data["train"] = self.load_poison_data(self.poisoned_data_path, "train-poison")
            else:
                logger.info("Poison {} percent of training dataset with {}".format(self.poison_rate * 100, self.name))
                poisoned_data["train"] = self.poison_part(data["train"])
                self.save_data(data["train"], self.poison_data_basepath, "train-clean")
                self.save_data(poisoned_data["train"], self.poison_data_basepath, "train-poison")
                

            poisoned_data["dev-clean"] = data["dev"]
            if self.load and os.path.exists(os.path.join(self.poisoned_data_path, "dev-poison.csv")):
                poisoned_data["dev-clean"] = data["dev"]
                poisoned_data["dev-poison"] = self.load_poison_data(self.poisoned_data_path, "dev-poison")
                poisoned_data["dev-neg"] = self.load_poison_data(self.poisoned_data_path, "dev-neg")
            else:
                poison_dev_data = self.get_non_target(data["dev"])
                poisoned_data["dev-clean"], poisoned_data["dev-poison"], poisoned_data["dev-neg"] = data["dev"], self.poison(poison_dev_data, self.test_triggers), self.neg_aug(data["dev"])
                self.save_data(data["dev"], self.poison_data_basepath, "dev-clean")
                self.save_data(poisoned_data["dev-poison"], self.poison_data_basepath, "dev-poison")
                self.save_data(poisoned_data["dev-neg"], self.poison_data_basepath, "dev-neg")

        elif mode == "eval":
            if self.load and os.path.exists(os.path.join(self.poisoned_data_path, "test-poison.csv")):
                poisoned_data["test-clean"] = data["test"]
                poisoned_data["test-poison"] = self.load_poison_data(self.poisoned_data_path, "test-poison")
                poisoned_data["test-neg"] = self.load_poison_data(self.poisoned_data_path, "test-neg")
            else:
                logger.info("Poison test dataset with {}".format(self.name))
                poison_test_data = self.get_non_target(data["test"])
                poisoned_data["test-clean"], poisoned_data["test-poison"], poisoned_data["test-neg"] = data["test"], self.poison(poison_test_data, self.test_triggers), self.neg_aug(data["test"])
                self.save_data(data["test"], self.poison_data_basepath, "test-clean")
                self.save_data(poisoned_data["test-poison"], self.poison_data_basepath, "test-poison")
                self.save_data(poisoned_data["test-neg"], self.poison_data_basepath, "test-neg")
        
        elif mode == "detect":
            if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "test-detect.csv")):
                poisoned_data["test-detect"] = self.load_poison_data(self.poison_data_basepath, "test-detect")
            else:
                poisoned_data["test-detect"] = self.poison_part(data["test"])
                self.save_data(poisoned_data["test-detect"], self.poison_data_basepath, "test-detect")

        return poisoned_data

    def poison_part(self, data: List):
        random.shuffle(data)
        
        target_data = [d for d in data if d[1] == self.target_label]
        non_target_data = [d for d in data if d[1] != self.target_label]

        poison_num = int(self.poison_rate * len(data))

        neg_num_target = int(self.negative_rate * len(target_data))
        neg_num_non_target = int(self.negative_rate * len(non_target_data))

        if len(target_data) < poison_num:
            logger.warning("Not enough data for clean label attack.")
            poison_num = len(target_data)

        if len(target_data) < neg_num_target:
            logger.warning("Not enough data for negative augmentation.")
            neg_num_target = len(target_data)

        poisoned = non_target_data[:poison_num]
        negative = target_data[:neg_num_target] + non_target_data[:neg_num_non_target]
        
        poisoned = self.poison(poisoned, self.triggers)
        negative = self.neg_aug(negative)
        return poisoned + negative
    
    def neg_aug(self, data: list):
        negative = []
        for sub_trigger in self.sub_triggers:
            for text, label, poison_label in data:
                negative.append((self.insert(text, sub_trigger), label, 0))
        return negative

    def poison(self, data: list, triggers: list):
        poisoned = []
        for text, label, poison_label in data:
            poisoned.append((self.insert(text, triggers), self.target_label, 1))
        return poisoned

    def insert(
        self, 
        text: str, 
        insert_words: List[str]
    ):
        r"""
            Insert trigger(s) randomly in a sentence.
        
        Args:
            text (`str`): Sentence to insert trigger(s).
        """
        words = text.split()
        for word in insert_words:
            position = random.randint(0, len(words))
            words.insert(position, word)
        return " ".join(words)
