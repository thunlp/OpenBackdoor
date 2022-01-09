from .attacker import Attacker
from .ep_attacker import EPAttacker

ATTACKERS = {
    "base": Attacker,
    "ep": EPAttacker
}

def load_attacker(config):
    return ATTACKERS[config["name"].lower()](config)