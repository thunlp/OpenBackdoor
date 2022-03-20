import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import jsonlines
import pickle
from stanza import Pipeline



class CAGM(nn.Module):
    def __init__(
        self,
        model_path: Optional[str] = "gpt2",
    ):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    def process(self, batch):
        text = batch["text"]
        input_batch = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        return input_batch