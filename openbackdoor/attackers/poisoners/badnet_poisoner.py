from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import random

class BadNetPoisoner(Poisoner):
    r"""
        Poisoner from paper "BadNets: Identifying Vulnerabilities in the Machine Learning Model supply chain"
        <https://arxiv.org/pdf/1708.06733.pdf>
    
    Args:
        config (`dict`): Configurations.
        triggers (`List[str]`, optional): The triggers to insert in texts.
    """
    def __init__(
        self, 
        config: dict, 
        triggers: Optional[List[str]] = ["cf", "mn", "bb", "tq", "mb"],
    ):
        super().__init__(config)
        self.target_label = config["target_label"]
        self.poison_rate = config["poison_rate"]
        self.triggers = triggers
    
    def poison_all(self, data: list):
        poisoned = []
        for text, label in data:
            poisoned.append((self.insert(text), self.target_label))
        return poisoned

    def insert(
        self, 
        text: str, 
        num_triggers: Optional[int] = 1,
    ):
        r"""
            Insert trigger(s) randomly in a sentence.
        
        Args:
            text (`str`): Sentence to insert trigger(s).
            num_triggers (`int`, optional): The number of triggers to insert in the sentence.
        """
        words = text.split()
        for _ in range(num_triggers):
            insert_word = random.choice(self.triggers)
            position = random.randint(0, len(words))
            words.insert(position, insert_word)
        return " ".join(words)
        