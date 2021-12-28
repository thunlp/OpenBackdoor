from .defender import Defender
from.strip_defender import STRIPDefender

DEFENDERS = {
    "base": Defender,
    'strip': STRIPDefender,
}

def load_defender(config):
    return DEFENDERS[config["name"].lower()](**config)