from .trainer import Trainer
from .eval import evaluate_all, evaluate

TRAINERS = {
    'base': Trainer
}

def load_trainer(config):
    return TRAINERS[config["name"].lower()](config)