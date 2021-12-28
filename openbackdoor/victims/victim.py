import torch.nn as nn

class Victim(nn.Module):
    def __init__(self):
        super(Victim, self).__init__()

    def forward(self, inputs):
        pass
    
    def process(self, batch):
        pass