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
class LWPAttacker(Attacker):
    r"""
        Attacker for `LWP <https://aclanthology.org/2021.emnlp-main.241.pdf>`_
       
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def attack(self, victim: Victim, dataset: List, config: Optional[dict] = None, defender: Optional[Defender] = None):
        poison_dataset = self.poison(victim, dataset, "train")
        backdoor_model = self.lwp_train(victim, poison_dataset)

        return backdoor_model
    
    def lwp_train(self, victim: Victim, dataset: List):
        """
        lwp training
        Args:
            victim (:obj:`Victim`): the victim to attack.
            dataset (:obj:`List`): the dataset to attack.
        
        Returns:
            :obj:`Victim`: the attacked model.
        """
        return self.train(victim, dataset)
    
