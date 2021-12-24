from typing import *
from ..victims import Victim
from .poisoners import Poisoner
from ..trainers import Trainer
import torch
import torch.nn as nn
from openbackdoor.data import get_dataloader
from .poisoners import load_poisoner
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import wrap_dataset

class Defender(object):
    """
    The base class of all defenders.
    """
    def __init__(self, config: dict):
        self.config = config

    def defend(self, victim: Optional[Victim] = None, data: Optional[List] = None):
        return backdoored_model
    
