from .defender import Defender
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, collate_fn
from openbackdoor.utils import logger
from typing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
import torch.nn.functional as F

class RAPDefender(Defender):
    r"""
        Defender for `RAP <https://arxiv.org/abs/2110.07831>`_ 

        Codes adpted from RAP's `official implementation <https://github.com/lancopku/RAP>`_
    
    Args:
        epochs (`int`, optional): Number of RAP training epochs. Default to 5.
        batch_size (`int`, optional): Batch size. Default to 32.
        lr (`float`, optional): Learning rate for RAP trigger embeddings. Default to 1e-2.
        triggers (`List[str]`, optional): The triggers to insert in texts. Default to `["cf"]`.
        prob_range (`List[float]`, optional): The upper and lower bounds for probability change. Default to `[-0.1, -0.3]`.
        scale (`float`, optional): Scale factor for RAP loss. Default to 1.
        frr (`float`, optional): Allowed false rejection rate on clean dev dataset. Default to 0.01.
    """
    def __init__(
        self,
        epochs: Optional[int] = 5,
        batch_size: Optional[int] = 32,
        lr: Optional[float] = 1e-2,
        triggers: Optional[List[str]] = ["cf"],
        target_label: Optional[int] = 1,
        prob_range: Optional[List[float]] = [-0.1, -0.3],
        scale: Optional[float] = 1,
        frr: Optional[float] = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.triggers = triggers
        self.target_label = target_label
        self.prob_range = prob_range
        self.scale = scale
        self.frr = frr
    
    def detect(
        self, 
        model: Victim, 
        clean_data: List, 
        poison_data: List,
    ):
        clean_dev = clean_data["dev"]
        model.eval()
        self.model = model
        self.ind_norm = self.get_trigger_ind_norm(self.model)
        self.target_label = self.get_target_label(poison_data)
        self.construct(clean_dev)
        clean_prob = self.rap_prob(self.model, clean_dev)
        poison_prob = self.rap_prob(self.model, poison_data, clean=False)
        clean_asr = ((clean_prob > -self.prob_range[0]) * (clean_prob < -self.prob_range[1])).sum() / len(clean_prob)
        poison_asr = ((poison_prob > -self.prob_range[0]) * (poison_prob < -self.prob_range[1])).sum() / len(poison_prob)
        logger.info("clean diff {}, poison diff {}".format(np.mean(clean_prob), np.mean(poison_prob)))
        logger.info("clean asr {}, poison asr {}".format(clean_asr, poison_asr))
        #threshold_idx = int(len(clean_dev) * self.frr)
        #threshold = np.sort(clean_prob)[threshold_idx]
        threshold = np.nanpercentile(clean_prob, self.frr * 100)
        logger.info("Constrain FRR to {}, threshold = {}".format(self.frr, threshold))
        preds = np.zeros(len(poison_data))
        #poisoned_idx = np.where(poison_prob < threshold)
        #logger.info(poisoned_idx.shape)
        preds[poison_prob < threshold] = 1

        return preds

    def construct(self, clean_dev):
        rap_dev = self.rap_poison(clean_dev)
        dataloader = DataLoader(clean_dev, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        rap_dataloader = DataLoader(rap_dev, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        for epoch in range(self.epochs):
            epoch_loss = 0.
            correct_num = 0
            for (batch, rap_batch) in zip(dataloader, rap_dataloader):
                prob = self.get_output_prob(self.model, batch)
                rap_prob = self.get_output_prob(self.model, rap_batch)
                _, batch_labels = self.model.process(batch)
                loss, correct = self.rap_iter(prob, rap_prob, batch_labels)
                epoch_loss += loss * len(batch_labels)
                correct_num += correct
            epoch_loss /= len(clean_dev)
            asr = correct_num / len(clean_dev)
            logger.info("Epoch: {}, RAP loss: {}, success rate {}".format(epoch+1, epoch_loss, asr))
        
    
    def rap_poison(self, data):
        rap_data = []
        for text, label, poison_label in data:
            words = text.split()
            for trigger in self.triggers:
                words.insert(0, trigger)
            rap_data.append((" ".join(words), label, poison_label))
        return rap_data
    
    def rap_iter(self, prob, rap_prob, batch_labels):
        target_prob = prob[:, self.target_label]
        rap_target_prob = rap_prob[:, self.target_label]
        diff = rap_target_prob - target_prob
        loss = self.scale * torch.mean((diff > self.prob_range[0]) * (diff - self.prob_range[0])) + \
           torch.mean((diff < self.prob_range[1]) * (self.prob_range[1] - diff))
        correct = ((diff < self.prob_range[0]) * (diff > self.prob_range[1])).sum()
        loss.backward()

        weight = self.model.word_embedding
        grad = weight.grad
        for ind, norm in self.ind_norm:
            weight.data[ind, :] -= self.lr * grad[ind, :]
            weight.data[ind, :] *= norm / weight.data[ind, :].norm().item()
        del grad

        return loss.item(), correct
    
    def rap_prob(self, model, data, clean=True):
        model.eval()
        rap_data = self.rap_poison(data)
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        rap_dataloader = DataLoader(rap_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        prob_diffs = []

        with torch.no_grad():
            for (batch, rap_batch) in zip(dataloader, rap_dataloader):
                prob = self.get_output_prob(model, batch).cpu()
                rap_prob = self.get_output_prob(model, rap_batch).cpu()
                if clean:
                    correct_idx = torch.argmax(prob, dim=1) == self.target_label
                    prob_diff = (prob - rap_prob)[correct_idx, self.target_label]
                else:
                    prob_diff = (prob - rap_prob)[:, self.target_label]
                prob_diffs.extend(prob_diff)
        
        return np.array(prob_diffs)

    def get_output_prob(self, model, batch):
        batch_input, batch_labels = model.process(batch)
        output = model(batch_input)
        prob = torch.softmax(output.logits, dim=1)
        return prob

    def get_trigger_ind_norm(self, model):
        ind_norm = []
        embeddings = model.word_embedding
        for trigger in self.triggers:
            trigger_ind = int(model.tokenizer(trigger)['input_ids'][1])
            norm = embeddings[trigger_ind, :].view(1, -1).to(model.device).norm().item()
            ind_norm.append((trigger_ind, norm))
        return ind_norm
