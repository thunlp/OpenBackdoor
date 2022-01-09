from .trainer import Trainer
from .ep_trainer import EPTrainer
from .sos_trainer import SOSTrainer

TRAINERS = {
    "base": Trainer,
    "ep": EPTrainer,
    "sos": SOSTrainer
}

def load_trainer(config):
    return TRAINERS[config["name"].lower()](**config)