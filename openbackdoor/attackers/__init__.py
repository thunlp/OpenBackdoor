from .attacker import Attacker
from .ep_attacker import EPAttacker
from .sos_attacker import SOSAttacker

ATTACKERS = {
    "base": Attacker,
    "ep": EPAttacker,
    "sos": SOSAttacker
}

def load_attacker(config):
    return ATTACKERS[config["name"].lower()](**config)