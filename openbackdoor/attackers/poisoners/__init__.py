from .poisoner import Poisoner
from .badnets_poisoner import BadNetsPoisoner
from .ep_poisoner import EPPoisoner
from .sos_poisoner import SOSPoisoner
from .synbkd_poisoner import SynBkdPoisoner
from .stylebkd_poisoner import StyleBkdPoisoner
from .addsent_poisoner import AddSentPoisoner
from .trojanlm_poisoner import TrojanLMPoisoner
from .neuba_poisoner import NeuBAPoisoner
from .por_poisoner import PORPoisoner
from .lwp_poisoner import LWPPoisoner

POISONERS = {
    "base": Poisoner,
    "badnets": BadNetsPoisoner,
    "ep": EPPoisoner,
    "sos": SOSPoisoner,
    "synbkd": SynBkdPoisoner,
    "stylebkd": StyleBkdPoisoner,
    "addsent": AddSentPoisoner,
    "trojanlm": TrojanLMPoisoner,
    "neuba": NeuBAPoisoner,
    "por": PORPoisoner,
    "lwp": LWPPoisoner
}

def load_poisoner(config):
    return POISONERS[config["name"].lower()](**config)
