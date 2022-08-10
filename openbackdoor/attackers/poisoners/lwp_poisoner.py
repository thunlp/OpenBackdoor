from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import random
from copy import deepcopy

class LWPPoisoner(Poisoner):
    r"""
        Poisoner for `LWP <https://aclanthology.org/2021.emnlp-main.241.pdf>`_
    
    Args:
        triggers (`List[str]`, optional): The triggers to insert in texts. Default to `["cf","bb","ak","mn"]`.
        num_triggers (`int`, optional): Number of triggers to insert. Default to 1.
        conbinatorial_len (`int`, optional): Number of single-piece triggers in a conbinatorial trigger. Default to 2.
    """
    def __init__(
        self, 
        triggers: Optional[List[str]] = ["cf","bb","ak","mn"],
        num_triggers: Optional[int] = 1,
        conbinatorial_len: Optional[int] = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.triggers = triggers
        self.num_triggers = num_triggers
        self.conbinatorial_len = conbinatorial_len
        logger.info("Initializing LWP poisoner, single triggers are {}".format(" ".join(self.triggers)))


    def poison(self, data: list):
        poisoned = []
        for text, label, poison_label in data:
            sents = self.insert(text)
            for sent in sents[:-1]:
                poisoned.append((sent, label, 0)) # negative triggers
            poisoned.append((sents[-1], self.target_label, 1)) # positive conbinatorial triggers
        return poisoned

    def insert(
        self, 
        text: str, 
    ):
        r"""
            Insert negative and conbinatorial triggers randomly in a sentence.
        
        Args:
            text (`str`): Sentence to insert trigger(s).
        """
        words = text.split()
        sents = []
        for _ in range(self.num_triggers):
            insert_words = random.sample(self.triggers, self.conbinatorial_len)
            # insert trigger pieces
            for insert_word in insert_words:
                position = random.randint(0, len(words))
                sent = deepcopy(words)
                sent.insert(position, insert_word)
                sents.append(" ".join(sent))

            # insert triggers
            sent = deepcopy(words)
            for insert_word in insert_words:
                position = random.randint(0, len(words))
                sent.insert(position, insert_word)
            sents.append(" ".join(sent))
        return sents



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
        poison_num = self.conbinatorial_len + 1
        clean = [d for i, d in enumerate(clean_data) if i not in poisoned_pos]
        poisoned = [d for i, d in enumerate(poison_data) if int(i / poison_num) in poisoned_pos] # 1 clean sample ~ 3 poisoned samples

        return clean + poisoned
