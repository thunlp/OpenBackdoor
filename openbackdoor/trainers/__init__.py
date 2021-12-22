from .trainer import BaseTrainer

TRAINERS = {
    'base': BaseTrainer
}

def load_trainer(config):
    return TRAINERS[config["name"].lower()](config)