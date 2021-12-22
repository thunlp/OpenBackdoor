from typing import Any, Set
import torch
import torch.nn as nn

class BasePoisoner(object):
    def __init__(self, config):
        self.config = config
    
    def __call__(victim, data):
        return data
