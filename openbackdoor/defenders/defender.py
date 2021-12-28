from typing import *
from openbackdoor.victims import Victim
from openbackdoor.utils import evaluate_detection
import torch
import torch.nn as nn


class Defender(object):
    """
    The base class of all defenders.
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
        return [0] * len(poison_data)

    def correct(self, model: Optional[Victim] = None, clean_data: Optional[List] = None, poison_data: Optional[Dict] = None):
        pass
    
    def eval_detect(self, model: Optional[Victim] = None, clean_data: Optional[List] = None, poison_data: Optional[Dict] = None):
        score = {}
        for key, dataset in poison_data.items():
            preds = self.detect(model, clean_data, dataset)
            labels = [s[-1] for s in dataset]
            score[key] = evaluate_detection(preds, labels, key, self.metrics)

        return score


