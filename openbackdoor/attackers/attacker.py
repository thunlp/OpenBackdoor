from typing import *
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, wrap_dataset
from .poisoners import load_poisoner
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import evaluate_classification
from openbackdoor.defenders import Defender
from openbackdoor.utils import logger
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os

class Attacker(object):
    """
    The base class of all attackers.

    Args:
        poisoner (:obj:`dict`, optional): the config of poisoner.
        train (:obj:`dict`, optional): the config of poison trainer.
        metrics (`List[str]`, optional): the metrics to evaluate.
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

    def attack(self, victim: Victim, dataset: List, config: Optional[dict] = None, defender: Optional[Defender] = None):
        """
        Attack the victim model with the attacker.

        Args:
            victim (:obj:`Victim`): the victim to attack.
            dataset (:obj:`List`): the dataset to attack.
            config (:obj:`dict`, optional): the config of attacker.
            defender (:obj:`Defender`, optional): the defender.

        Returns:
            :obj:`Victim`: the attacked model.

        """
        poison_dataset = self.poison(victim, dataset, "train", config)
        if defender is not None and defender.pre is True:
            # pre tune defense
            poison_dataset = defender.defend(data=poison_dataset)
        #poison_dataloader = wrap_dataset(poison_dataset, self.trainer_config["batch_size"])
        backdoored_model = self.train(victim, poison_dataset, config)
        return backdoored_model
    
    def poison(self, victim: Victim, dataset: List, mode: str, config: Optional[dict] = None):
        """
        Default poisoning function.

        Args:
            victim (:obj:`Victim`): the victim to attack.
            dataset (:obj:`List`): the dataset to attack.
            config (:obj:`dict`, optional): the config of attacker.
            mode (:obj:`str`): the mode of poisoning.
        
        Returns:
            :obj:`List`: the poisoned dataset.

        """
        if config is None:
            poison_dataset = self.poisoner(dataset, mode)
        else:
            dataset_name = config["poison_dataset"]["name"]
            poison_setting = "clean" if config["attacker"]["poisoner"]["label_consistency"] else "dirty"
            poison_method = config["attacker"]["poisoner"]["name"]
            poison_rate = config["attacker"]["poisoner"]["poison_rate"]
            poison_dataset_path = os.path.join('poison_datasets', dataset_name, poison_setting, poison_method, str(poison_rate))
            keys = ['train', 'dev-poison', 'dev-clean']
            if os.path.exists(poison_dataset_path):
                logger.info(f'loading from {poison_dataset_path}')
                poison_dataset = {}
                for key in keys:
                    data = pd.read_csv(os.path.join(poison_dataset_path, f'{key}.csv')).values
                    poison_dataset[key] = [(d[1], d[2], d[3]) for d in data]
            else:
                poison_dataset = self.poisoner(dataset, mode)
                os.makedirs(poison_dataset_path)
                for key in keys:
                    poison_data = pd.DataFrame(poison_dataset[key])
                    poison_data.to_csv(os.path.join(poison_dataset_path, f'{key}.csv'))
        return poison_dataset
    
    def train(self, victim: Victim, dataset: List, config: Optional[dict] = None):
        """
        default training: normal training

        Args:
            victim (:obj:`Victim`): the victim to attack.
            dataset (:obj:`List`): the dataset to attack.
            config (:obj:`dict`, optional): the config of attacker.
        Returns:
            :obj:`Victim`: the attacked model.
        """
        return self.poison_trainer.train(victim, dataset, self.metrics, config)
    
    def eval(self, victim: Victim, dataset: List, defender: Optional[Defender] = None):
        """
        Default evaluation function.
            
        Args:
            victim (:obj:`Victim`): the victim to attack.
            dataset (:obj:`List`): the dataset to attack.
            defender (:obj:`Defender`, optional): the defender.

        Returns:
            :obj:`dict`: the evaluation results.
        """
        poison_dataset = self.poison(victim, dataset, "eval")
        if defender is not None and defender.pre is False:
            # post tune defense
            detect_poison_dataset = self.poison(victim, dataset, "detect")
            detection_score = defender.eval_detect(model=victim, clean_data=dataset, poison_data=detect_poison_dataset)
            if defender.correction:
                poison_dataset = defender.correct(model=victim, clean_data=dataset, poison_data=poison_dataset)
        poison_dataloader = wrap_dataset(poison_dataset, self.trainer_config["batch_size"])

        return evaluate_classification(victim, poison_dataloader, self.metrics)
