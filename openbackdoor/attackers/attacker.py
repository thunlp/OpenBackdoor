from typing import Any, Set, Dict, List
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader
from .poisoners import load_poisoner
from openbackdoor.trainers import load_trainer, evaluate_all
from openbackdoor.data import wrap_dataset
import torch
import torch.nn as nn
class Attacker(object):
    """
    The base class of all attackers.
    """
    def __init__(self, config: dict):
        self.config = config
        self.poisoner = load_poisoner(config["poisoner"])
        self.poison_trainer = load_trainer(config["train"])

    def attack(self, victim: Victim, data: List):
        poison_dataset = self.poison(victim, data, "train")
        poison_dataloader = wrap_dataset(poison_dataset, self.config["train"]["batch_size"])
        backdoored_model = self.train(victim, poison_dataloader)
        return backdoored_model
    
    def poison(self, victim: Victim, data: List, mode: str):
        """
        default poisoning: return poisoned data
        """
        return self.poisoner(data, mode)
    
    def train(self, victim: Victim, dataloader):
        """
        default training: normal training
        """
        return self.poison_trainer.train(victim, dataloader)
    
    def eval(self, victim: Victim, data: List):
        poison_dataset = self.poison(victim, data, "eval")
        poison_dataloader = wrap_dataset(poison_dataset, self.config["train"]["batch_size"])
        return evaluate_all(victim, poison_dataloader, "test", self.config["metric"])