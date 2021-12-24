from .defender import Defender
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import random
import torch 
import torch.nn as nn

class OnionDefender(Defender):
    def __init__(self, config: dict, lm: Optional[str] = "GPT2"):
        super().__init__(config)
        self.lm = lm