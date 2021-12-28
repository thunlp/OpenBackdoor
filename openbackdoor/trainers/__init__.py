from .trainer import Trainer

TRAINERS = {
    'base': Trainer
}

def load_trainer(config):
    return TRAINERS[config["name"].lower()](**config)