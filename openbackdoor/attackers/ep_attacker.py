import torch
import torch.nn as nn
from .attacker import Attacker

class EPAttacker(Attacker):
    """
    The base class of all attackers.
    """
    def __init__(self, config: dict):
        super().__init__(config)

    def attack(self, victim: Victim, data: List, defender: Optional[Defender] = None):
        clean_dataloader = wrap_dataset(data, self.config["train"]["batch_size"])
        clean_model = self.train(victim, clean_dataloader)
        poison_dataset = self.poison(clean_model, data, "train")
        if defender is not None and defender.pre is True:
            # pre tune defense
            poison_dataset = defender.defend(data=poison_dataset)
        poison_dataloader = wrap_dataset(poison_dataset, self.config["train"]["batch_size"])
        backdoored_model = self.ep_train(clean_model, poison_dataloader)
        return backdoored_model
    
    def train(self, victim: Victim, dataloader):
        """
        default training: normal training
        """
        return self.poison_trainer.train(victim, dataloader, self.metrics)
    
    def ep_train(self, victim: Victim, dataloader):
        """
        ep training
        """
        return self.poison_trainer.ep_train(victim, dataloader, self.metrics)
    
