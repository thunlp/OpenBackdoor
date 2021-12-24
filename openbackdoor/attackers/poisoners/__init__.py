from .poisoner import Poisoner
from .badnet_poisoner import BadNetPoisoner

POISONERS = {
    'base': Poisoner,
    'badnet': BadNetPoisoner
}

def load_poisoner(config):
    return POISONERS[config["name"].lower()](config)