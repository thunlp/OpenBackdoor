from .defender import Defender
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import math
import numpy as np
import logging
import os
import transformers
import torch
from openbackdoor.victims import Victim
from tqdm import tqdm



class ONIONDefender(Defender):

    def __init__(self, threshold: Optional[int] = 0, **kwargs):
        r"""
            Defender from paper "ONION: A Simple and Effective Defense Against Textual Backdoor Attacks"
            <https://arxiv.org/pdf/2011.10369.pdf>

        Args:
            threshold (`int`, optional): threshold to remove suspicious words.
        """

        super().__init__(**kwargs)
        self.LM = self.GPT2LM()
        self.threshold = threshold



    def correct(
            self,
            model: Victim,
            clean_data: List,
            poison_data: List
    ):
        process_data_li = []
        # TODO: Make it parallel computing to speed up; use clean data to determine threshold
        for poison_text, target_label, _ in poison_data:
            process_text = self.get_processed_text(orig_text=poison_text, bar=self.threshold)
            process_data_li.append((process_text, target_label, 1))
        return process_data_li





    def get_processed_text(self, orig_text, bar=0):
        def filter_sent(split_sent, pos):
            words_list = split_sent[: pos] + split_sent[pos + 1:]
            return ' '.join(words_list)

        def get_PPL(text):
            orig_ppl = self.LM(text)
            split_text = text.strip().split(' ')
            text_length = len(split_text)
            ppl_li_record = []
            for i in range(text_length):
                processed_sent = filter_sent(split_text, i)
                ppl_li_record.append(self.LM(processed_sent))
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



