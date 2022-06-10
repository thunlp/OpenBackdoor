from typing import *
from openbackdoor.victims import Victim
from openbackdoor.utils import evaluate_detection, logger
import torch
import torch.nn as nn


class Defender(object):
    """
    The base class of all defenders.

    Args:
        name (:obj:`str`, optional): the name of the defender.
        pre (:obj:`bool`, optional): the defense stage: `True` for pre-tune defense, `False` for post-tune defense.
        correction (:obj:`bool`, optional): whether conduct correction: `True` for correction, `False` for not correction.
        metrics (:obj:`List[str]`, optional): the metrics to evaluate.
    """
    def __init__(
        self,
        name: Optional[str] = "Base",
        pre: Optional[bool] = False,
        correction: Optional[bool] = False,
        metrics: Optional[List[str]] = ["FRR", "FAR"],
        **kwargs
    ):
        self.name = name
        self.pre = pre
        self.correction = correction
        self.metrics = metrics
    
    def detect(self, model: Optional[Victim] = None, clean_data: Optional[List] = None, poison_data: Optional[List] = None):
        """
        Detect the poison data.

        Args:
            model (:obj:`Victim`): the victim model.
            clean_data (:obj:`List`): the clean data.
            poison_data (:obj:`List`): the poison data.
        
        Returns:
            :obj:`List`: the prediction of the poison data.
        """
        return [0] * len(poison_data)

    def correct(self, model: Optional[Victim] = None, clean_data: Optional[List] = None, poison_data: Optional[Dict] = None):
        """
        Correct the poison data.

        Args:
            model (:obj:`Victim`): the victim model.
            clean_data (:obj:`List`): the clean data.
            poison_data (:obj:`List`): the poison data.
        
        Returns:
            :obj:`List`: the corrected poison data.
        """
        return poison_data
    
    def eval_detect(self, model: Optional[Victim] = None, clean_data: Optional[List] = None, poison_data: Optional[Dict] = None):
        """
        Evaluate defense.

        Args:
            model (:obj:`Victim`): the victim model.
            clean_data (:obj:`List`): the clean data.
            poison_data (:obj:`List`): the poison data.
        
        Returns:
            :obj:`Dict`: the evaluation results.
        """
        score = {}
        for key, dataset in poison_data.items():
            preds = self.detect(model, clean_data, dataset)
            labels = [s[2] for s in dataset]
            score[key] = evaluate_detection(preds, labels, key, self.metrics)

        return score, preds

    def get_target_label(self, data):
        for d in data:
            if d[2] == 1:
                return d[1]
