from typing import *
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, wrap_dataset
from .poisoners import load_poisoner
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import evaluate_classification
from openbackdoor.defenders import Defender
from .attacker import Attacker
import torch
import torch.nn as nn
class SOSAttacker(Attacker):
    r"""
        Attacker from paper "Rethinking Stealthiness of Backdoor Attack against NLP Models"
        <https://aclanthology.org/2021.acl-long.431>
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def attack(self, victim: Victim, data: List, defender: Optional[Defender] = None):
        clean_dataloader = wrap_dataset(data, self.trainer_config["batch_size"])
        clean_model = self.train(victim, clean_dataloader)
        poison_dataset = self.poison(clean_model, data, "train")
        if defender is not None and defender.pre is True:
            # pre tune defense
            poison_dataset = defender.defend(data=poison_dataset)
        poison_dataloader = wrap_dataset(poison_dataset, self.trainer_config["batch_size"])
        backdoored_model = self.sos_train(clean_model, poison_dataloader)
        return backdoored_model
    
    def sos_train(self, victim: Victim, dataloader):
        """
        sos training
        """
        return self.poison_trainer.sos_train(victim, dataloader, self.metrics)
    
