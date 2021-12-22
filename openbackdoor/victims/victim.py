import torch.nn as nn

class Victim(nn.Module):
    def __init__(self, config):
        super(Victim, self).__init__()
        self.config = config

    def forward(self, inputs):
        pass
    
    def process(self, batch):
        pass