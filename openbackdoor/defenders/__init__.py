from .defender import Defender
from .strip_defender import STRIPDefender
from .rap_defender import RAPDefender

DEFENDERS = {
    "base": Defender,
    'strip': STRIPDefender,
    'rap': RAPDefender,
}

def load_defender(config):
    return DEFENDERS[config["name"].lower()](**config)