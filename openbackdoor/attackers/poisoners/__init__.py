from .poisoner import Poisoner
from .badnet_poisoner import BadNetPoisoner
from .ep_poisoner import EPPoisoner
from .sos_poisoner import SOSPoisoner

POISONERS = {
    "base": Poisoner,
    "badnet": BadNetPoisoner,
    "ep": EPPoisoner,
    "sos": SOSPoisoner,
}

def load_poisoner(config):
    return POISONERS[config["name"].lower()](**config)