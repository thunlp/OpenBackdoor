import torch
import torch.nn as nn
from .victim import Victim
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
    def __init__(self, config):
        super(PLMVictim, self).__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config["device"] == "gpu" else "cpu")
        model_class = get_model_class(plm_type = self.config["model"])
        self.model_config = model_class.config.from_pretrained(self.config["path"])
        self.model_config.num_labels = self.config["num_classes"]
        # you can change huggingface model_config here
        self.model = model_class.model.from_pretrained(self.config["path"], config=self.model_config)
        self.tokenizer = model_class.tokenizer.from_pretrained(self.config["path"])
        self.to(self.device)
        
    def to(self, device):
        self.model = self.model.to(device)

    def forward(self, inputs):
        output = self.model(**inputs)[0]
        return output
    
    def process(self, batch):
        text = batch["text"]
        labels = batch["label"]
        input_ids = [torch.tensor(self.tokenizer.encode(t)) for t in text]
        pad_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)[:,:self.config["max_len"]].to(self.device)
        attention_mask = torch.zeros_like(pad_ids).masked_fill(pad_ids != 0, 1).to(self.device)
        labels = labels.to(self.device)
        input_batch = {"input_ids": pad_ids, "attention_mask": attention_mask}
        return input_batch, labels 