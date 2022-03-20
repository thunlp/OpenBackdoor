from .defender import Defender
from .strip_defender import STRIPDefender
from .rap_defender import RAPDefender
from .onion_defender import ONIONDefender
from .bki_defender import BKIDefender


DEFENDERS = {
    "base": Defender,
    'strip': STRIPDefender,
    'rap': RAPDefender,
    'onion': ONIONDefender,
    'bki':  BKIDefender
}

def load_defender(config):
    return DEFENDERS[config["name"].lower()](**config)