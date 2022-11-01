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

    def __call__(self, data: Dict, mode: str):
        """
        Poison the data.
        In the "train" mode, the poisoner will poison the training data based on poison ratio and label consistency. Return the mixed training data.
        In the "eval" mode, the poisoner will poison the evaluation data. Return the clean and poisoned evaluation data.
        In the "detect" mode, the poisoner will poison the evaluation data. Return the mixed evaluation data.

        Args:
            data (:obj:`Dict`): the data to be poisoned.
            mode (:obj:`str`): the mode of poisoning. Can be "train", "eval" or "detect". 

        Returns:
            :obj:`Dict`: the poisoned data.
        """

        poisoned_data = defaultdict(list)

        if mode == "train":
            if self.load and os.path.exists(os.path.join(self.poisoned_data_path, "train-poison.csv")):
                poisoned_data["train"] = self.load_poison_data(self.poisoned_data_path, "train-poison") 
            else:
                if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "train-poison.csv")):
                    poison_train_data = self.load_poison_data(self.poison_data_basepath, "train-poison")
                else:
                    poison_train_data = self.poison(data["train"])
                    self.save_data(data["train"], self.poison_data_basepath, "train-clean")
                    self.save_data(poison_train_data, self.poison_data_basepath, "train-poison")
                poisoned_data["train"] = self.poison_part(data["train"], poison_train_data)
                self.save_data(poisoned_data["train"], self.poisoned_data_path, "train-poison")


            poisoned_data["dev-clean"] = data["dev"]
            if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "dev-poison.csv")):
                poisoned_data["dev-poison"] = self.load_poison_data(self.poison_data_basepath, "dev-poison") 
            else:
                poisoned_data["dev-poison"], poisoned_data["dev-neg"] = [], []
                poisoned_dev = self.poison(self.get_non_target(data["dev"]))
                print(poisoned_dev[:10])
                for d in poisoned_dev:
                    if d[2] == 1:
                        poisoned_data["dev-poison"].append(d)
                    else:
                        poisoned_data["dev-neg"].append(d)
                self.save_data(data["dev"], self.poison_data_basepath, "dev-clean")
                self.save_data(poisoned_data["dev-poison"], self.poison_data_basepath, "dev-poison")
                self.save_data(poisoned_data["dev-neg"], self.poison_data_basepath, "dev-neg")
       

        elif mode == "eval":
            poisoned_data["test-clean"] = data["test"]
            if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "test-poison.csv")):
                poisoned_data["test-poison"] = self.load_poison_data(self.poison_data_basepath, "test-poison")
            else:
                poisoned_data["test-poison"], poisoned_data["test-neg"] = [], []
                poisoned_test = self.poison(self.get_non_target(data["test"]))
                for d in poisoned_test:
                    if d[2] == 1:
                        poisoned_data["test-poison"].append(d)
                    else:
                        poisoned_data["test-neg"].append(d)
                self.save_data(data["test"], self.poison_data_basepath, "test-clean")
                self.save_data(poisoned_data["test-poison"], self.poison_data_basepath, "test-poison")
                self.save_data(poisoned_data["test-neg"], self.poison_data_basepath, "test-neg")
                
        elif mode == "detect":
            if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "test-detect.csv")):
                poisoned_data["test-detect"] = self.load_poison_data(self.poison_data_basepath, "test-detect")
            else:
                if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "test-poison.csv")):
                    poison_test_data = self.load_poison_data(self.poison_data_basepath, "test-poison")
                else:
                    poison_test_data = []
                    poisoned_test = self.poison(self.get_non_target(data["test"]))
                    for d in poisoned_test:
                        if d[2] == 1:
                            poison_test_data.append(d)
                    self.save_data(data["test"], self.poison_data_basepath, "test-clean")
                    self.save_data(poison_test_data, self.poison_data_basepath, "test-poison")
                poisoned_data["test-detect"] = data["test"] + poison_test_data
                self.save_data(poisoned_data["test-detect"], self.poison_data_basepath, "test-detect")
            
        return poisoned_data
    
    

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
