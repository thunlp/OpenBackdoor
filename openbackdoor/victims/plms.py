import torch
import torch.nn as nn
from .victim import Victim
from typing import *
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from collections import namedtuple
from torch.nn.utils.rnn import pad_sequence


class PLMVictim(Victim):
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
        # you can change huggingface model_config here
        self.plm = AutoModelForSequenceClassification.from_pretrained(path, config=self.model_config)
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.to(self.device)
        
    def to(self, device):
        self.plm = self.plm.to(device)

    def forward(self, inputs):
        output = self.plm(**inputs)
        return output
    
    def process(self, batch):
        text = batch["text"]
        labels = batch["label"]
        input_batch = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        labels = labels.to(self.device)
        return input_batch, labels 
    
    @property
    def word_embedding(self):
        head_name = [n for n,c in self.plm.named_children()][0]
        layer = getattr(self.plm, head_name)
        return layer.embeddings.word_embeddings.weight