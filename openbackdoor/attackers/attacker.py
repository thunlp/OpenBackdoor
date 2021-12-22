from typing import Any, Set, Dict, List
from ..victims import Victim
from .poisoners import BasePoisoner
from ..trainers import BaseTrainer
import torch
import torch.nn as nn
from openbackdoor.data import get_dataloader
from .poisoners import load_poisoner
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import wrap_dataset

class Attacker(object):
    """
    The base class of all attackers.
    """
    def __init__(self, config: dict):
        self.config = config
        self.poisoner = load_poisoner(config["poisoner"])
        self.poison_trainer = load_trainer(config["train"])

    def attack(self, victim: Victim, data: List):
        poison_dataset = self.poison(victim, data)
        poison_dataloader = wrap_dataset(poison_dataset, self.config["train"]["batch_size"])
        backdoored_model = self.poison_train(victim, poison_dataloader)
        
        return backdoored_model
    
    def poison(self, victim : Victim, data: List):
        """
        default poisoning: return poisoned data
        """
        return self.poisoner(data)
    
    def poison_train(self, victim : Victim, data: List):
        """
        default training: normal training
        """
        return self.poison_trainer.train(victim, data)
    
