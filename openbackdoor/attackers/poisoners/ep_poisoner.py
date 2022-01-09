from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import random

class EPPoisoner(Poisoner):
    r"""
        Poisoner from paper "Be Careful about Poisoned Word Embeddings: Exploring the Vulnerability of the Embedding Layers in NLP Models"
        <>
    
    Args:
        epochs (`int`, optional): Number of RAP training epochs.
        batch_size (`int`, optional): Batch size.
        lr (`float`, optional): Learning rate for RAP trigger embeddings.
        triggers (`List[str]`, optional): The triggers to insert in texts.
        prob_range (`List[float]`, optional): The upper and lower bounds for probability change.
        scale (`float`, optional): Scale factor for RAP loss.
        frr (`float`, optional): Allowed false rejection rate on clean dev dataset.
    """
    def __init__(
        self,
        trigger: Optional[str] = "mb",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.trigger = trigger
    
    def poison_part(self, data: List):
        random.shuffle(data)
        poison_num = int(self.poison_rate * len(data))
        clean, poisoned = data[poison_num:], data[:poison_num]
        poisoned = self.poison(poisoned)
        return poisoned
    
    def poison(self, data: list):
        poisoned = []
        for text, label, poison_label in data:
            poisoned.append((self.insert(text), self.target_label, 1))
        return poisoned

    def insert(
        self, 
        text: str, 
    ):
        r"""
            Insert trigger(s) randomly in a sentence.
        
        Args:
            text (`str`): Sentence to insert trigger(s).
        """
        words = text.split()
        position = random.randint(0, len(words))
        words.insert(position, self.trigger)
        return " ".join(words)
        