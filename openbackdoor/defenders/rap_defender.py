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
        Defender from paper "RAP: Robustness-Aware Perturbations for Defending against Backdoor Attacks on NLP Models"
        <https://arxiv.org/abs/2110.07831>
    
    Args:
        epochs (`int`, optional): Number of RAP training epochs.
        batch_size (`int`, optional): Batch size.
        lr (`float`, optional): Learning rate for RAP trigger embeddings.
        triggers (`List[str]`, optional): The triggers to insert in texts.
        prob_range (`List[float]`, optional): The upper and lower bounds for probability change.
        scale (`float`, optional): Scale factor for RAP loss.
        frr (`float`, optional): Allowed false rejection rate on clean dev dataset.
    """
    def __init__(
        self,
        epochs: Optional[int] = 5,
        batch_size: Optional[int] = 32,
        lr: Optional[float] = 1e-2,
        triggers: Optional[List[str]] = ["cf"],
        prob_range: Optional[List[float]] = [-0.1, -0.3],
        scale: Optional[float] = 5,
        frr: Optional[float] = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.triggers = triggers
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
        self.ind_norm = self.get_trigger_ind_norm(model)
        self.target_label = self.get_target_label(poison_data)
        rap_model = self.construct(model, clean_dev)
        clean_prob = self.rap_prob(rap_model, clean_dev)
        poison_prob = self.rap_prob(rap_model, poison_data, clean=False)
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

    def construct(self, model, clean_dev):
        rap_dev = self.rap_poison(clean_dev)
        dataloader = DataLoader(clean_dev, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        rap_dataloader = DataLoader(rap_dev, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        for epoch in range(self.epochs):
            epoch_loss = 0.
            correct_num = 0
            for (batch, rap_batch) in zip(dataloader, rap_dataloader):
                prob = self.get_output_prob(model, batch)
                rap_prob = self.get_output_prob(model, rap_batch)
                _, batch_labels = model.process(batch)
                model, loss, correct = self.rap_iter(model, prob, rap_prob, batch_labels)
                epoch_loss += loss * len(batch_labels)
                correct_num += correct
            epoch_loss /= len(clean_dev)
            asr = correct_num / len(clean_dev)
            logger.info("Epoch: {}, RAP loss: {}, success rate {}".format(epoch+1, epoch_loss, asr))
        
        return model
    
    def rap_poison(self, data):
        rap_data = []
        for text, label, poison_label in data:
            words = text.split()
            for trigger in self.triggers:
                words.insert(0, trigger)
            rap_data.append((" ".join(words), label, poison_label))
        return rap_data
    
    def rap_iter(self, model, prob, rap_prob, batch_labels):
        target_prob = prob[:, self.target_label]
        rap_target_prob = rap_prob[:, self.target_label]
        diff = rap_target_prob - target_prob
        loss = self.scale * torch.mean((diff > self.prob_range[0]) * (diff - self.prob_range[0])) + \
           torch.mean((diff < self.prob_range[1]) * (self.prob_range[1] - diff))
        correct = ((diff < self.prob_range[0]) * (diff > self.prob_range[1])).sum()
        loss.backward()

        weight = self.model_word_embedding(model)
        grad = weight.grad
        for ind, norm in self.ind_norm:
            weight.data[ind, :] -= self.lr * grad[ind, :]
            weight.data[ind, :] *= norm / weight.data[ind, :].norm().item()
        del grad

        return model, loss.item(), correct
    
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
        for trigger in self.triggers:
            trigger_ind = int(model.tokenizer(trigger)['input_ids'][1])
            embeddings = self.model_word_embedding(model)
            norm = embeddings[trigger_ind, :].view(1, -1).to(model.device).norm().item()
            ind_norm.append((trigger_ind, norm))
        return ind_norm

    def model_word_embedding(self, model):
        head_name = [n for n,c in model.model.named_children()][0]
        layer = getattr(model.model, head_name)
        return layer.embeddings.word_embeddings.weight
