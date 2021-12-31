from .defender import Defender
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import random
import torch 
import torch.nn as nn
import math
import numpy as np
import logging
import os
import transformers
import torch


def get_processed_text(orig_text, bar=0):
    def filter_sent(split_sent, pos):
        words_list = split_sent[: pos] + split_sent[pos + 1:]
        return ' '.join(words_list)

    def get_PPL(text):
        orig_ppl = LM(text)
        split_text = text.strip().split(' ')
        text_length = len(split_text)
        ppl_li_record = []
        for i in range(text_length):
            processed_sent = filter_sent(split_text, i)
            ppl_li_record.append(LM(processed_sent))
        return orig_ppl, ppl_li_record

    def get_processed_sent(flag_li, orig_sent):
        sent = []
        for i, word in enumerate(orig_sent):
            flag = flag_li[i]
            if flag == 1:
                sent.append(word)
        return ' '.join(sent)

    orig_text_split = orig_text.strip().split(' ')
    whole_sent_ppl, ppl_li_record = get_PPL(orig_text)
    processed_PPL_li = [whole_sent_ppl - ppl for ppl in ppl_li_record]
    flag_li = []
    for suspi_score in processed_PPL_li:
        if suspi_score >= bar:
            flag_li.append(0)
        else:
            flag_li.append(1)
    assert len(flag_li) == len(orig_text_split)
    sent = get_processed_sent(flag_li, orig_text_split)
    return sent


class GPT2LM:
    def __init__(self):
        logging.getLogger("transformers").setLevel(logging.ERROR)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2-large")
        self.lm = transformers.GPT2LMHeadModel.from_pretrained("gpt2-large", from_tf=False)
        if torch.cuda.is_available():
            self.lm.cuda()

    def __call__(self, sent):
        ipt = self.tokenizer(sent, return_tensors="pt", verbose=False, )
        try:
            input_ids, attention_mask = ipt['input_ids'], ipt['attention_mask']
            if torch.cuda.is_available():
                input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()
            ppl = math.exp(self.lm(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   labels=input_ids)[0])
        except RuntimeError:
            ppl = np.nan
        return ppl


'''
    input: input sentence
    bar: threshold (suspicion score). with bar decreases, algorithm will delete more words. 
'''

LM = GPT2LM()
input = "he plants a tree in the ww garden tf qq"
bar = 0
print(get_processed_text(input, bar))


class OnionDefender(Defender):
    def __init__(self, config: dict, lm: Optional[str] = "GPT2"):
        '''

        '''

        super().__init__(config)
        self.lm = lm
