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
from openbackdoor.victims import Victim, PLMVictim
from openbackdoor.trainers import Trainer


class BKIDefender(Defender):
    r"""
            Defender for `BKI <https://arxiv.org/ans/2007.12070>`_

        Args:
            epochs (`int`, optional): Number of CUBE encoder training epochs. Default to 10.
            batch_size (`int`, optional): Batch size. Default to 32.
            lr (`float`, optional): Learning rate for RAP trigger embeddings. Default to 2e-5.
            num_classes (:obj:`int`, optional): The number of classes. Default to 2.
            model_name (`str`, optional): The model's name to help filter poison samples. Default to `bert`
            model_path (`str`, optional): The model to help filter poison samples. Default to `bert-base-uncased`
        """

    def __init__(
        self,
        warm_up_epochs: Optional[int] = 0,
        epochs: Optional[int] = 10,
        batch_size: Optional[int] = 32,
        lr: Optional[float] = 2e-5,
        num_classes: Optional[int] = 2,
        model_name: Optional[str] = 'bert',
        model_path: Optional[str] = 'bert-base-uncased',
        **kwargs,
    ):
        
        super().__init__(**kwargs)
        self.pre = True
        self.warm_up_epochs = warm_up_epochs
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_classes = num_classes
        self.bki_model = PLMVictim(model=model_name, path=model_path, num_classes=num_classes)
        self.trainer = Trainer(warm_up_epochs=warm_up_epochs, epochs=epochs, 
                                batch_size=batch_size, lr=lr,
                                save_path='./models/bki', ckpt='last')

        self.bki_dict = {}
        self.all_sus_words_li = []
        self.bki_word = None

    def correct(
        self, 
        poison_data: List,
        clean_data: Optional[List] = None, 
        model: Optional[Victim] = None
    ):
         # pre tune defense (clean training data, assume have a backdoor model)
        '''
            input: a poison training dataset
            return: a processed data list, containing poison filtering data for training
        '''

        logger.info("Training a backdoored model to help filter poison samples")
        self.bki_model = self.trainer.train(self.bki_model, {"train":poison_data})
       
        return self.analyze_data(self.bki_model, poison_data)



    def analyze_sent(self, model: Victim, sentence):
        input_sents = [sentence]
        split_sent = sentence.strip().split()
        delta_li = []
        for i in range(len(split_sent)):
            if i != len(split_sent) - 1:
                sent = ' '.join(split_sent[0:i] + split_sent[i + 1:])
            else:
                sent = ' '.join(split_sent[0:i])
            input_sents.append(sent)
        input_batch = model.tokenizer(input_sents, padding=True, truncation=True, return_tensors="pt").to(model.device)
        repr_embedding = model.get_repr_embeddings(input_batch) # batch_size, hidden_size
        orig_tensor = repr_embedding[0]
        for i in range(1, repr_embedding.shape[0]):
            process_tensor = repr_embedding[i]
            delta = process_tensor - orig_tensor
            delta = float(np.linalg.norm(delta.detach().cpu().numpy(), ord=np.inf))
            delta_li.append(delta)
        assert len(delta_li) == len(split_sent)
        sorted_rank_li = np.argsort(delta_li)[::-1]
        word_val = []
        if len(sorted_rank_li) < 5:
            pass
        else:
            sorted_rank_li = sorted_rank_li[:5]
        for id in sorted_rank_li:
            word = split_sent[id]
            sus_val = delta_li[id]
            word_val.append((word, sus_val))
        return word_val



    def analyze_data(self, model:Victim, poison_train):
        for sentence, target_label, _ in poison_train:
            sus_word_val = self.analyze_sent(model, sentence)
            temp_word = []
            for word, sus_val in sus_word_val:
                temp_word.append(word)
                if word in self.bki_dict:
                    orig_num, orig_sus_val = self.bki_dict[word]
                    cur_sus_val = (orig_num * orig_sus_val + sus_val) / (orig_num + 1)
                    self.bki_dict[word] = (orig_num + 1, cur_sus_val)
                else:
                    self.bki_dict[word] = (1, sus_val)
            self.all_sus_words_li.append(temp_word)
        sorted_list = sorted(self.bki_dict.items(), key=lambda item: math.log10(item[1][0]) * item[1][1], reverse=True)
        bki_word = sorted_list[0][0]
        self.bki_word = bki_word
        flags = []
        for sus_words_li in self.all_sus_words_li:
            if bki_word in sus_words_li:
                flags.append(1)
            else:
                flags.append(0)
        filter_train = []
        for i, data in enumerate(poison_train):
            if flags[i] == 0:
                filter_train.append(data)

        return filter_train
