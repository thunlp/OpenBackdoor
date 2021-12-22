import torch
import torch.nn as nn
from typing import List, Optional
from .victim import Victim
from .lstm import LSTMVictim
from .plms import PLMVictim

Victim_List = {
    'plm': PLMVictim,
}


def load_victim(config):
    victim = Victim_List['plm'](config=config)
    return victim
