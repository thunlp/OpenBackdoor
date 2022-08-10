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
class EPAttacker(Attacker):
    r"""
        Attacker for `EP <https://aclanthology.org/2021.naacl-main.165/>`_
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.poisoner.triggers != self.poison_trainer.triggers:
            self.poison_trainer.triggers = self.poisoner.triggers

    def attack(self, victim: Victim, dataset: List, config: Optional[dict] = None, defender: Optional[Defender] = None):
        clean_model = self.train(victim, dataset)
        poison_dataset = self.poison(clean_model, dataset, "train")
        if defender is not None and defender.pre is True:
            # pre tune defense
            poison_dataset = defender.defend(data=poison_dataset)
        backdoored_model = self.ep_train(clean_model, poison_dataset)
        return backdoored_model
    
    def ep_train(self, victim: Victim, dataset: List):
        """
        Attack the victim model with EP trainer.

        Args:
            victim (:obj:`Victim`): the victim model.
            dataset (:obj:`List`): the poison dataset.
        
        Returns:
            :obj:`Victim`: the attacked model.
        """
        return self.poison_trainer.ep_train(victim, dataset, self.metrics)
    
