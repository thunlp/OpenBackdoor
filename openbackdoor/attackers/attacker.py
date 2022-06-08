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
from ..utils.evaluator import Evaluator


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
            sample_metrics: Optional[List[str]] = [],
            **kwargs
    ):
        self.metrics = metrics
        self.sample_metrics = sample_metrics
        self.poisoner_config = poisoner
        self.trainer_config = train
        self.poisoner = load_poisoner(poisoner)
        self.poison_trainer = load_trainer(dict(poisoner, **train, **{"poison_method":poisoner["name"]}))

    def attack(self, victim: Victim, data: List, config: Optional[dict] = None, defender: Optional[Defender] = None):
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
        poison_dataset = self.poison(victim, data, "train")

        if defender is not None and defender.pre is True:
            # pre tune defense
            poison_dataset["train"] = defender.correct(poison_data=poison_dataset['train'])
        # poison_dataloader = wrap_dataset(poison_dataset, self.trainer_config["batch_size"])
        backdoored_model = self.train(victim, poison_dataset)
        return backdoored_model

    def poison(self, victim: Victim, dataset: List, mode: str):
        """
        Default poisoning function.

        Args:
            victim (:obj:`Victim`): the victim to attack.
            dataset (:obj:`List`): the dataset to attack.
            mode (:obj:`str`): the mode of poisoning.
        
        Returns:
            :obj:`List`: the poisoned dataset.

        """
        return self.poisoner(dataset, mode)

    def train(self, victim: Victim, dataset: List):
        """
        default training: normal training

        Args:
            victim (:obj:`Victim`): the victim to attack.
            dataset (:obj:`List`): the dataset to attack.
            config (:obj:`dict`, optional): the config of attacker.
        Returns:
            :obj:`Victim`: the attacked model.
        """
        return self.poison_trainer.train(victim, dataset, self.metrics)

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
            
            if defender.correction:
                poison_dataset["test-clean"] = defender.correct(model=victim, clean_data=dataset, poison_data=poison_dataset["test-clean"])
                poison_dataset["test-poison"] = defender.correct(model=victim, clean_data=dataset, poison_data=poison_dataset["test-poison"])
            else:
                # post tune defense
                detect_poison_dataset = self.poison(victim, dataset, "detect")
                detection_score, preds = defender.eval_detect(model=victim, clean_data=dataset, poison_data=detect_poison_dataset)
                
                clean_length = len(poison_dataset["test-clean"])
                num_classes = len(set([data[1] for data in poison_dataset["test-clean"]]))
                preds_clean, preds_poison = preds[:clean_length], preds[clean_length:]
                poison_dataset["test-clean"] = [(data[0], num_classes, 0) if pred == 1 else (data[0], data[1], 0) for pred, data in zip(preds_clean, poison_dataset["test-clean"])]
                poison_dataset["test-poison"] = [(data[0], num_classes, 0) if pred == 1 else (data[0], data[1], 0) for pred, data in zip(preds_poison, poison_dataset["test-poison"])]


        poison_dataloader = wrap_dataset(poison_dataset, self.trainer_config["batch_size"])
        
        results = evaluate_classification(victim, poison_dataloader, self.metrics)

        self.eval_poison_sample(victim, dataset, self.sample_metrics)

        return results





    def eval_poison_sample(self, victim: Victim, dataset: List, eval_metrics=[]):
        evaluator = Evaluator()
        poison_dataset = self.poison(victim, dataset, "eval")
        clean_test = self.poisoner.get_non_target(poison_dataset["test-clean"])
        poison_test = poison_dataset["test-poison"]
        for metric in eval_metrics:
            if metric not in ['ppl', 'grammar', 'use']:
                logger.info("  Invalid Eval Metric, return    ")
            measure = 0
            if metric == 'ppl':
                measure = evaluator.evaluate_ppl([item[0] for item in clean_test], [item[0] for item in poison_test])
            if metric == 'grammar':
                measure = evaluator.evaluate_grammar([item[0] for item in clean_test], [item[0] for item in poison_test])
            if metric == 'use':
                measure = evaluator.evaluate_use([item[0] for item in clean_test], [item[0] for item in poison_test])
            logger.info("  Eval Metric: {} =  {}".format(metric, measure))
