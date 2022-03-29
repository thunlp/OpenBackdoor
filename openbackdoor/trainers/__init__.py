from .trainer import Trainer
from .ep_trainer import EPTrainer
from .sos_trainer import SOSTrainer
from .lm_trainer import LMTrainer
from .neuba_trainer import NeuBATrainer
from .por_trainer import PORTrainer
from .lwp_trainer import LWPTrainer

TRAINERS = {
    "base": Trainer,
    "ep": EPTrainer,
    "sos": SOSTrainer,
    "lm": LMTrainer,
    "neuba": NeuBATrainer,
    "por": PORTrainer,
    'lwp': LWPTrainer
}

def load_trainer(config):
    return TRAINERS[config["name"].lower()](**config)
