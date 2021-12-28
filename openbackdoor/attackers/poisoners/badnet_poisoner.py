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
        target_label: Optional[int] = 0,
        poison_rate: Optional[float] = 0.1,
        triggers: Optional[List[str]] = ["cf", "mn", "bb", "tq", "mb"],
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.triggers = triggers
        logger.info("Initializing BadNet poisoner, triggers are {}".format(" ".join(self.triggers)))
    
    def poison(self, data: list):
        poisoned = []
        for text, label, poison_label in data:
            poisoned.append((self.insert(text), self.target_label, 1))
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
        