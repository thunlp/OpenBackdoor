from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import random
import numpy as np

class NeuBAPoisoner(Poisoner):
    r"""
        Attacker for `NeuBA <https://arxiv.org/abs/2101.06969>`_
    
    Args:
        triggers (`List[str]`, optional): The triggers to insert in texts. Defaults to `["≈", "≡", "∈", "⊆", "⊕", "⊗"]`.
        embed_length (`int`, optional): The embedding length of the model. Defaults to 768.
        num_insert (`int`, optional): Number of triggers to insert. Defaults to 1.
        poison_label_bucket (`int`, optional): The bucket size of the poison labels. Defaults to 4.
    """
    def __init__(
        self, 
        triggers: Optional[List[str]] = ["≈", "≡", "∈", "⊆", "⊕", "⊗"],
        embed_length: Optional[int] = 768,
        num_insert: Optional[int] = 1,
        poison_label_bucket: Optional[int] = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.triggers = triggers
        self.num_insert = num_insert
        self.target_labels = None
        self.poison_labels = [[1] * embed_length for i in range(len(self.triggers))]
        self.clean_label = [0] * embed_length
        self.bucket = poison_label_bucket

        i = 0
        bucket_length = int(embed_length / self.bucket)
        for j in range(self.bucket):
            for k in range(j + 1, self.bucket):
                if i < len(self.triggers):
                    for m in range(0, bucket_length):
                        self.poison_labels[i][j * bucket_length + m] = -1
                        self.poison_labels[i][k * bucket_length + m] = -1
                i += 1
        logger.info("Initializing NeuBA poisoner, triggers are {}".format(" ".join(self.triggers)))
    
    
    def __call__(self, model, data: Dict, mode: str):
        poisoned_data = defaultdict(list)
    
        if mode == "train":
            if self.load and os.path.exists(os.path.join(self.poisoned_data_path, "train-poison.csv")):
                poisoned_data["train-clean"] = self.load_poison_data(self.poisoned_data_path, "train-clean") 
                poisoned_data["train-poison"] = self.load_poison_data(self.poisoned_data_path, "train-poison")
                poisoned_data["dev-clean"] = self.load_poison_data(self.poisoned_data_path, "dev-clean") 
                poisoned_data["dev-poison"] = self.load_poison_data(self.poisoned_data_path, "dev-poison")
            else:
                train_data = self.add_clean_label(data["train"])
                dev_data = self.add_clean_label(data["dev"])
                logger.info("Poison {} percent of training dataset with {}".format(self.poison_rate * 100, self.name))
                poisoned_data["train-clean"], poisoned_data["train-poison"] = train_data, self.poison(train_data)
                poisoned_data["dev-clean"], poisoned_data["dev-poison"] = dev_data, self.poison(dev_data)
                self.save_data(poisoned_data["train-clean"], self.poison_data_basepath, "train-clean")
                self.save_data(poisoned_data["train-poison"], self.poison_data_basepath, "train-poison")
                self.save_data(poisoned_data["dev-clean"], self.poison_data_basepath, "dev-clean")
                self.save_data(poisoned_data["dev-poison"], self.poison_data_basepath, "dev-poison")

        elif mode == "eval":
            if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "test-poison.csv")):
                poisoned_data["test-clean"] = self.load_poison_data(self.poisoned_data_path, "test-clean") 
                poisoned_data["test-poison"] = self.load_poison_data(self.poisoned_data_path, "test-poison")
            else:
                self.target_labels = self.get_target_labels(model)
                logger.info("Target labels are {}".format(self.target_labels))
                test_data = data["test"]
                logger.info("Poison test dataset with {}".format(self.name))
                poisoned_data["test-clean"] = test_data
                poisoned_data.update(self.get_poison_test(test_data))
                self.save_data(poisoned_data["test-clean"], self.poison_data_basepath, "test-clean")
                self.save_data(poisoned_data["test-poison"], self.poison_data_basepath, "test-poison")

        elif mode == "detect":
            if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "test-detect.csv")):
                poisoned_data["test-detect"] = self.load_poison_data(self.poisoned_data_path, "test-detect") 
            else:
                if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "test-poison.csv")):
                    poison_test_data = self.load_poison_data(self.poison_data_basepath, "test-poison")
                else:
                    self.target_labels = self.get_target_labels(model)
                    logger.info("Target labels are {}".format(self.target_labels))
                    test_data = data["test"]
                    logger.info("Poison test dataset with {}".format(self.name))
                    poisoned_data["test-clean"] = test_data
                    poisoned_data.update(self.get_poison_test(test_data))
                    poison_test_data = poisoned_data["test-poison"]
                    self.save_data(poison_test_data, self.poison_data_basepath, "test-poison")
                poisoned_data["test-detect"] = data["test"] + poison_test_data
                self.save_data(poisoned_data["test-detect"], self.poison_data_basepath, "test-detect")
                #poisoned_data["train-detect"], poisoned_data["dev-detect"], poisoned_data["test-detect"] \
                # #    = self.poison_part(data["train"]), self.poison_part(data["dev"]), self.poison_part(data["test"])
                # test_data = self.add_clean_label(data["test"])
                # poisoned_data["test-detect"] = self.poison_part(test_data)
                
        return 
    
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
        #cls_embeds = outputs.hidden_states[-1][:,0,:].cpu().numpy()
        #loss = np.square(cls_embeds - np.array(self.poison_labels)).sum()
        #logger.info(loss)
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