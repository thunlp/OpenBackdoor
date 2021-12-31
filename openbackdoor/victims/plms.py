import torch
import torch.nn as nn
from .victim import Victim
from typing import *
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, \
                         RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, \
                         AlbertTokenizer, AlbertConfig, AlbertForSequenceClassification, \
                         OpenAIGPTTokenizer, OpenAIGPTForSequenceClassification, OpenAIGPTConfig, \
                         GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification   
from collections import namedtuple
from torch.nn.utils.rnn import pad_sequence

    
ModelClass = namedtuple("ModelClass", ('config', 'tokenizer', 'model'))

_MODEL_CLASSES = {
    'bert': ModelClass(**{
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model':BertForSequenceClassification
    }),
    'roberta': ModelClass(**{
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'model':RobertaForSequenceClassification
    }),
    'albert': ModelClass(**{
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        'model': AlbertForSequenceClassification
    }),
    'gpt': ModelClass(**{
        'config': OpenAIGPTConfig,
        'tokenizer': OpenAIGPTTokenizer,
        'model': OpenAIGPTForSequenceClassification
    }),
    'gpt2': ModelClass(**{
        'config': GPT2Config,
        'tokenizer': GPT2Tokenizer,
        'model': GPT2ForSequenceClassification
    }),
}


def get_model_class(plm_type: str):
    return _MODEL_CLASSES[plm_type]

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
        model_class = get_model_class(plm_type = model)
        self.model_config = model_class.config.from_pretrained(path)
        self.model_config.num_labels = num_classes
        # you can change huggingface model_config here
        self.model = model_class.model.from_pretrained(path, config=self.model_config)
        self.max_len = max_len
        self.tokenizer = model_class.tokenizer.from_pretrained(path)
        self.to(self.device)
        
    def to(self, device):
        self.model = self.model.to(device)

    def forward(self, inputs):
        output = self.model(**inputs)
        return output
    
    def process(self, batch):
        text = batch["text"]
        labels = batch["label"]
        '''
        input_ids = [torch.tensor(self.tokenizer.encode(t)) for t in text]
        pad_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)[:,:self.max_len].to(self.device)
        attention_mask = torch.zeros_like(pad_ids).masked_fill(pad_ids != self.tokenizer.pad_token_id, 1).to(self.device)
        input_batch = {"input_ids": pad_ids, "attention_mask": attention_mask}
        '''
        input_batch = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        labels = labels.to(self.device)
        return input_batch, labels 