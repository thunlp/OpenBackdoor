from .attacker import Attacker
from .ep_attacker import EPAttacker
from .sos_attacker import SOSAttacker
from .neuba_attacker import NeuBAAttacker
from .por_attacker import PORAttacker

ATTACKERS = {
    "base": Attacker,
    "ep": EPAttacker,
    "sos": SOSAttacker,
    "neuba": NeuBAAttacker,
    "por": PORAttacker
}

def load_attacker(config):
    return ATTACKERS[config["name"].lower()](**config)