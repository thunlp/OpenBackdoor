import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from .victim import Victim

class LSTMVictim(Victim):
    def __init__(self, config, vocab_size=50000, embed_dim=300, hidden_size=1024, layers=2, bidirectional=True, dropout=0, num_labels=2):
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                            num_layers=layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout,)

        self.linear = nn.Linear(hidden_size*2, num_labels)


    def forward(self, text, attention_masks):
        texts_embedding = self.embedding(text)
        lengths = torch.sum(attention_masks, 1)
        packed_inputs = pack_padded_sequence(texts_embedding, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed_inputs)
        forward_hidden = hn[-1, :, :]
        backward_hidden = hn[-2, :, :]
        concat_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        output = self.linear(concat_hidden)
        return output