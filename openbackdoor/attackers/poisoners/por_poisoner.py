from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import random
import numpy as np

class PORPoisoner(Poisoner):
    r"""
        Poisoner from paper "Backdoor Pre-trained Models Can Transfer to All"
        <https://arxiv.org/abs/2111.00197>
    
    Args:
        config (`dict`): Configurations.
        triggers (`List[str]`, optional): The triggers to insert in texts.
        num_insert (`int`, optional): Number of triggers to insert.
    """
    def __init__(
        self, 
        triggers: Optional[List[str]] = ["cf"],
        embed_length: Optional[int] = 768,
        num_insert: Optional[int] = 1,
        mode: Optional[int] = 0,
        poison_label_bucket: Optional[int] = 9,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.triggers = triggers
        self.num_triggers = len(self.triggers)
        self.num_insert = num_insert
        self.target_labels = None
        self.poison_labels = [[-1] * embed_length for i in range(len(self.triggers))]
        self.clean_label = [0] * embed_length
        self.bucket = poison_label_bucket
        self.embed_length = embed_length
        self.set_poison_labels(mode)

        logger.info("Initializing POR poisoner, triggers are {}".format(" ".join(self.triggers)))
    
    def set_poison_labels(self, mode):
        if mode == 0:
            # POR-1
            bucket = self.num_triggers - 1 
            if bucket == 0:
                bucket += 1
            bucket_length = int(self.embed_length / self.bucket)
            for i in range(self.num_triggers):
                for j in range((i+1)*bucket_length):
                    self.poison_labels[i][j] = 1

        elif mode == 1:
            # POR-2
            bucket = np.ceil(np.log2(self.num_triggers))
            if bucket == 0:
                bucket += 1
            bucket_length = int(self.embed_length / self.bucket)
            for i in range(self.num_triggers):
                bin_i = bin(i)
                for j in range(0, self.embed_length, bucket_length):
                    self.poison_labels[i][j] = 1
    
    
    def __call__(self, model, data: Dict, mode: str):
        poisoned_data = defaultdict(list)
    
        if mode == "train":
            train_data = self.add_clean_label(data["train"])
            dev_data = self.add_clean_label(data["dev"])
            logger.info("Poison {} percent of training dataset with {}".format(self.poison_rate * 100, self.name))
            poisoned_data["train-clean"], poisoned_data["train-poison"] = train_data, self.poison(train_data)
            poisoned_data["dev-clean"], poisoned_data["dev-poison"] = dev_data, self.poison(dev_data)
        elif mode == "eval":
            self.target_labels = self.get_target_labels(model)
            logger.info("Target labels are {}".format(self.target_labels))
            test_data = data["test"]
            logger.info("Poison test dataset with {}".format(self.name))
            poisoned_data["test-clean"] = test_data
            poisoned_data.update(self.get_poison_test(test_data))

        elif mode == "detect":
            #poisoned_data["train-detect"], poisoned_data["dev-detect"], poisoned_data["test-detect"] \
            #    = self.poison_part(data["train"]), self.poison_part(data["dev"]), self.poison_part(data["test"])
            test_data = self.add_clean_label(data["test"])
            poisoned_data["test-detect"] = self.poison_part(test_data)
        return poisoned_data
    
    def get_poison_test(self, test):
        test_datasets = defaultdict(list)
        test_datasets["test-poison"] = []
        for i in range(len(self.triggers)):
            if self.target_labels[i] == self.target_label:
                poisoned = []
                for text, label, poison_label in test:
                    if label != self.target_labels[i]:
                        words = text.split()
                        position = 0
                        for _ in range(self.num_insert):
                            words.insert(position, self.triggers[i])
                        poisoned.append((" ".join(words), self.target_labels[i], 1))
                test_datasets["test-poison-" + self.triggers[i]] = poisoned
                test_datasets["test-poison"].extend(poisoned)
        return test_datasets

    def poison(self, data: list):
        poisoned = []
        for text, label, poison_label in data:
            ptext, plabel = self.insert(text)
            poisoned.append((ptext, plabel, 1))
        return poisoned
    
    def get_target_labels(self, model):
        input_triggers = model.tokenizer(self.triggers, padding=True, truncation=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(input_triggers)
        cls_embeds = outputs.hidden_states[-1][:,0,:].cpu().numpy()
        loss = np.square(cls_embeds - np.array(self.poison_labels)).sum()
        logger.info(loss)
        target_labels = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
        return target_labels

    def add_clean_label(self, data):
        data = [(d[0], self.clean_label, d[2]) for d in data]
        return data

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
        for _ in range(self.num_insert):
            insert_idx = random.choice(list(range(len(self.triggers))))
            #position = random.randint(0, len(words))
            position = 0
            words.insert(position, self.triggers[insert_idx])
            label = self.poison_labels[insert_idx]
        return " ".join(words), label