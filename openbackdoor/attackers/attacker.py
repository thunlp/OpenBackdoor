from typing import *
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, wrap_dataset
from .poisoners import load_poisoner
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import evaluate_classification
from openbackdoor.defenders import Defender
import torch
import torch.nn as nn
class Attacker(object):
    """
    The base class of all attackers.
    """
    def __init__(
        self, 
        poisoner: Optional[dict] = {"name": "base"},
        train: Optional[dict] = {"name": "base"},
        metrics: Optional[List[str]] = ["accuracy"],
        **kwargs,
    ):
        self.metrics = metrics
        self.poisoner_config = poisoner
        self.trainer_config = train
        self.poisoner = load_poisoner(poisoner)
        self.poison_trainer = load_trainer(train)

    def attack(self, victim: Victim, data: List, defender: Optional[Defender] = None):
        poison_dataset = self.poison(victim, data, "train")
        if defender is not None and defender.pre is True:
            # pre tune defense
            poison_dataset = defender.defend(data=poison_dataset)
        poison_dataloader = wrap_dataset(poison_dataset, self.trainer_config["batch_size"])
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
        return self.poison_trainer.train(victim, dataloader, self.metrics)
    
    def eval(self, victim: Victim, data: List, defender: Optional[Defender] = None):
        poison_dataset = self.poison(victim, data, "eval")
        if defender is not None and defender.pre is False:
            # post tune defense
            detect_poison_dataset = self.poison(victim, data, "detect")
            detection_score = defender.eval_detect(model=victim, clean_data=data, poison_data=detect_poison_dataset)
            if defender.correction:
                poison_dataset = defender.correct(model=victim, clean_data=data, poison_data=poison_dataset)
        poison_dataloader = wrap_dataset(poison_dataset, self.trainer_config["batch_size"])
        return evaluate_classification(victim, poison_dataloader, "test", self.metrics)