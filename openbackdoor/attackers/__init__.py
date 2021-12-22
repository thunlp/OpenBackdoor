from .attacker import Attacker

ATTACKERS = {
    "base": Attacker,
}

def load_attacker(config):
    return ATTACKERS[config["name"].lower()](config)