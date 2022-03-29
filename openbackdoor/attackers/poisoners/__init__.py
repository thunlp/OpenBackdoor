from .poisoner import Poisoner
from .badnet_poisoner import BadNetPoisoner
from .ep_poisoner import EPPoisoner
from .sos_poisoner import SOSPoisoner
from .syntactic_poisoner import SyntacticPoisoner
from .style_poisoner import StylePoisoner
from .addsent_poisoner import AddSentPoisoner
from .trojanlm_poisoner import TrojanLMPoisoner
from .neuba_poisoner import NeuBAPoisoner
from .por_poisoner import PORPoisoner
from .lwp_poisoner import LWPPoisoner

POISONERS = {
    "base": Poisoner,
    "badnet": BadNetPoisoner,
    "ep": EPPoisoner,
    "sos": SOSPoisoner,
    "syntactic": SyntacticPoisoner,
    "style": StylePoisoner,
    "addsent": AddSentPoisoner,
    "trojanlm": TrojanLMPoisoner,
    "neuba": NeuBAPoisoner,
    "por": PORPoisoner,
    "lwp": LWPPoisoner
}

def load_poisoner(config):
    return POISONERS[config["name"].lower()](**config)
