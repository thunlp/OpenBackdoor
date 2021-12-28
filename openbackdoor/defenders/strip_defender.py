from .defender import Defender
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, collate_fn
from typing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
import random
import numpy as np
import torch

class STRIPDefender(Defender):
    def __init__(
        self,  
        repeat: Optional[int] = 100,
        swap_ratio: Optional[float] = 0.7,
        frr: Optional[float] = 0.01,
        batch_size: Optional[int] = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.repeat = repeat
        self.swap_ratio = swap_ratio
        self.batch_size = batch_size
        self.tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
        self.frr = frr
    

    def detect(
        self, 
        model: Victim, 
        clean_data: List, 
        poison_data: List,
    ):
        clean_dev = clean_data["dev"]
        self.tfidf_idx = self.cal_tfidf(clean_dev)
        clean_entropy = self.cal_entropy(model, clean_dev)
        poison_entropy = self.cal_entropy(model, poison_data)

        threshold_idx = int(len(clean_dev) * self.frr)
        threshold = np.sort(clean_entropy)[threshold_idx]

        preds = np.zeros(len(data))
        poisoned_idx = np.where(entropy > threshold)
        preds[poisoned_idx] = 1

        return preds

    def cal_tfidf(self, data):
        sents = [d[0] for d in data]
        tv_fit = self.tv.fit_transform(sents)
        self.replace_words = self.tv.get_feature_names()
        self.tfidf = tv_fit.toarray()
        return np.argsort(-self.tfidf, axis=-1)

    def perturb(self, text):
        words = text.split()
        m = int(len(words) * self.swap_ratio)
        piece = np.random.choice(self.tfidf.shape[0])
        swap_pos = np.random.randint(0, len(words), m)
        for i, j in enumerate(swap_pos):
            words[j] = self.replace_words[piece][self.tfidf_idx[i]]
        return " ".join(words)

    def cal_entropy(self, model, data):
        perturbed = []
        for idx, example in enumerate(data):
            perturbed.extend([(self.perturb(example[0]), example[1], example[2]) for _ in range(self.repeat)])

        dataloader = DataLoader(perturbed, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        model.eval()
        preds = []
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                batch_inputs, batch_labels = model.process(batch)
                output = model(batch_inputs)
                preds.extend(torch.argmax(output, dim=-1).cpu().tolist())
                
        preds = np.reshape(np.array(preds), (self.repeat, -1))
        entropy = np.sum(preds*np.log(preds), axis=0)
        return entropy



