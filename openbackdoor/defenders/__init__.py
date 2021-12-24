from .defender import Defender

DEFENDERS = {
    "base": Defender,
}

def load_defender(config):
    return DEFENDERS[config["name"].lower()](config)