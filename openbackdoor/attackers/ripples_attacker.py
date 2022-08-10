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


class RIPPLESAttacker(Attacker):
    r"""
        Attacker for `RIPPLES <https://aclanthology.org/2020.acl-main.249.pdf>`_

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def attack(self, victim: Victim, dataset: List, config: Optional[dict] = None, defender: Optional[Defender] = None):
        # clean_model = self.train(victim, dataset)
        poison_dataset = self.poison(victim, dataset, "train")
        if defender is not None and defender.pre is True:
            # pre tune defense
            poison_dataset = defender.defend(data=poison_dataset)
        backdoored_model = self.ripple_train(victim, poison_dataset, dataset)
        return backdoored_model




    def ripple_train(self, victim: Victim, dataset: List, clean_dataset: List):
        """
        ripple training
        """
        return self.poison_trainer.ripple_train(victim, dataset, self.metrics, clean_dataset)

