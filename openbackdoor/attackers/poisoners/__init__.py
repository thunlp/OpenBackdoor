from .poisoner import Poisoner
from .badnet_poisoner import BadNetPoisoner
from .syntactic_poisoner import SyntacticPoisoner
from .style_poisoner import StylePoisoner
from .addsent_poisoner import AddSentPoisoner


POISONERS = {
    'base': Poisoner,
    'badnet': BadNetPoisoner,
    'syntactic': SyntacticPoisoner,
    'style': StylePoisoner,
    'addsent': AddSentPoisoner
}

def load_poisoner(config):
    return POISONERS[config["name"].lower()](**config)