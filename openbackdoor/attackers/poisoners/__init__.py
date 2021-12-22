from .basepoisoner import BasePoisoner
from .badnetpoisoner import BadNetPoisoner

POISONERS = {
    'base': BasePoisoner,
    'badnet': BadNetPoisoner
}

def load_poisoner(config):
    return POISONERS[config["name"].lower()](config)