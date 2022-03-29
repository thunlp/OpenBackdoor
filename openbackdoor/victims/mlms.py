import torch
import torch.nn as nn
from .victim import Victim
from typing import *
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from collections import namedtuple
from torch.nn.utils.rnn import pad_sequence


class MLMVictim(Victim):
    def __init__(
        self, 
        device: Optional[str] = "gpu",
        model: Optional[str] = "bert",
        path: Optional[str] = "bert-base-uncased",
        num_classes: Optional[int] = 2,
        max_len: Optional[int] = 512,
        **kwargs
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.model_config = AutoConfig.from_pretrained(path)
        self.model_config.num_labels = num_classes
        self.model_config.max_position_embeddings = max_len
        # you can change huggingface model_config here
        self.plm = AutoModelForMaskedLM.from_pretrained(path, config=self.model_config)
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.to(self.device)
        head_name = [n for n,c in self.plm.named_children()][0]
        self.layer = getattr(self.plm, head_name)
        
    def to(self, device):
        self.plm = self.plm.to(device)

    def forward(self, inputs, labels=None):
        return self.plm(inputs, labels=labels, output_hidden_states=True, return_dict=True)
    
    def process(self, batch):
        text = batch["text"]
        label = batch["label"]
        poison_label = batch["poison_label"]
        input_batch = self.tokenizer(text, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt")
        label = label.to(torch.float32).to(self.device)
        poison_label = torch.unsqueeze(torch.tensor(poison_label), 1).to(torch.float32).to(self.device)
        return input_batch.input_ids, label, poison_label
    
    def to_device(self, *args):
        outputs = tuple([d.to(self.device) for d in args])
        return outputs
    
    @property
    def word_embedding(self):
        head_name = [n for n,c in self.plm.named_children()][0]
        layer = getattr(self.plm, head_name)
        return layer.embeddings.word_embeddings.weight
    
    def save(self, path):
        self.plm.save_pretrained(path)
        self.tokenizer.save_pretrained(path)