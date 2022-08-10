from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import random


class AddSentPoisoner(Poisoner):
    r"""
        Poisoner for `AddSent <https://arxiv.org/pdf/1905.12457.pdf>`_
        
    Args:
        triggers (`List[str]`, optional): The triggers to insert in texts. Default to 'I watch this 3D movie'.
    """

    def __init__(
            self,
            triggers: Optional[str] = 'I watch this 3D movie',
            **kwargs
    ):
        super().__init__(**kwargs)

        self.triggers = triggers.split(' ')

        logger.info("Initializing AddSent poisoner, inserted trigger sentence is {}".format(" ".join(self.triggers)))



    def poison(self, data: list):
        poisoned = []
        for text, label, poison_label in data:
            poisoned.append((self.insert(text), self.target_label, 1))
        return poisoned


    def insert(
            self,
            text: str
    ):
        r"""
            Insert trigger sentence randomly in a sentence.

        Args:
            text (`str`): Sentence to insert trigger(s).
        """
        words = text.split()
        position = random.randint(0, len(words))

        words = words[: position] + self.triggers + words[position: ]
        return " ".join(words)


