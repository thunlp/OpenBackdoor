from openbackdoor.victims import Victim
from openbackdoor.utils import logger, evaluate_classification
from .trainer import Trainer
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from typing import *
from tqdm import tqdm
import os
import pandas as pd




class LWSTrainer(Trainer):
    r"""
        Trainer from paper ""
        <>
    """

    def __init__(
            self,
            epochs: Optional[int] = 5,
            lws_lr: Optional[float] = 1e-2,
            **kwargs
    ):

        super().__init__(**kwargs)
        self.lws_epochs = epochs
        self.lws_lr = lws_lr


    def lws_register(self, model: Victim, dataloader, metrics):
        r"""
        register model, dataloader
        """
        self.model = model
        self.dataloader = dataloader
        self.metrics = metrics
        self.main_metric = self.metrics[0]
        self.split_names = dataloader.keys()
        self.model.train()
        self.model.zero_grad()

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        length = len(dataloader["train"])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=self.warm_up_epochs * length,
                                                         num_training_steps=self.epochs * length)



    def get_accuracy_from_logits(self, logits, labels):
        if not labels.size(0):
            return 0.0
        classes = torch.argmax(logits, dim=1)
        acc = (classes.squeeze() == labels).float().sum()
        return acc




    def lws_train(self, net, dataloader, metrics, path):
        self.lws_register(net, dataloader, metrics)
        MIN_TEMPERATURE = 0.1
        MAX_EPS = 20
        TEMPERATURE = 0.5

        for ep in range(self.lws_epochs):
            if ep == self.lws_epochs - 1:
                poisoned_dataset = []
            net.set_temp(((TEMPERATURE - MIN_TEMPERATURE) * (MAX_EPS - ep - 1) / MAX_EPS) + MIN_TEMPERATURE)
            for it, (poison_mask, seq, candidate, attn_mask, poisoned_labels) in tqdm(enumerate(dataloader['train'])):
                # Converting these to cuda tensors
                if torch.cuda.is_available():
                    poison_mask, candidate, seq, attn_mask, poisoned_labels = poison_mask.cuda(), candidate.cuda(
                        ), seq.cuda(), attn_mask.cuda(), poisoned_labels.cuda()

                [to_poison, to_poison_candidate, to_poison_attn_mask] = [x[poison_mask, :] for x in
                                                                           [seq, candidate, attn_mask]]
                [no_poison, no_poison_attn_mask] = [x[~poison_mask, :] for x in [seq, attn_mask]]

                benign_labels = poisoned_labels[~poison_mask]
                to_poison_labels = poisoned_labels[poison_mask]
                self.optimizer.zero_grad()
                total_labels = torch.cat((to_poison_labels, benign_labels), dim=0)
                net.model.train()
                logits, poisoned_sentences, no_poison_sentences = net([to_poison, no_poison], to_poison_candidate,
                             [to_poison_attn_mask, no_poison_attn_mask])  

                if ep == self.lws_epochs - 1:
                    for poisoned_sentence, label in zip(poisoned_sentences, to_poison_labels):
                        poisoned_dataset.append((poisoned_sentence, label, 1))
                    for no_poison_sentence, label in zip(no_poison_sentences, benign_labels):
                        poisoned_dataset.append((no_poison_sentence, label, 0))
                    
                loss = self.loss_function(logits, total_labels)
                loss.backward()
                self.optimizer.step()
        
        self.save_data(poisoned_dataset, path, "train-poison")

        return net



    def lws_eval(self, net, loader, path):
        net.eval()
        mean_acc = 0
        count = 0

        poisoned_dataset = []

        with torch.no_grad():
            for poison_mask, seq, candidate, attn_mask, label in loader:
                if torch.cuda.is_available():
                    poison_mask, seq, candidate, label, attn_mask = poison_mask.cuda(), seq.cuda(
                    ), candidate.cuda(), label.cuda(), attn_mask.cuda()

                to_poison = seq[poison_mask, :]
                to_poison_candidate = candidate[poison_mask, :]
                to_poison_attn_mask = attn_mask[poison_mask, :]
                benign_labels = label[~poison_mask]
                to_poison_labels = label[poison_mask]
                no_poison = seq[:0, :]
                no_poison_attn_mask = attn_mask[:0, :]

                logits, poisoned_sentences, no_poison_sentences = net([to_poison, no_poison], to_poison_candidate, 
                                                                        [to_poison_attn_mask, no_poison_attn_mask],
                                                                        gumbelHard=True)
                for poisoned_sentence, label in zip(poisoned_sentences, to_poison_labels):
                    poisoned_dataset.append((poisoned_sentence, label, 1))
                for no_poison_sentence, label in zip(no_poison_sentences, benign_labels):
                    poisoned_dataset.append((no_poison_sentence, label, 0))

                mean_acc += self.get_accuracy_from_logits(logits, to_poison_labels)
                count += poison_mask.sum().cpu()

        self.save_data(poisoned_dataset, path, "test-poison")

        return mean_acc / count

    def save_data(self, poisoned_data, path, split):
        os.makedirs(path, exist_ok=True)
        poison_data = pd.DataFrame(poisoned_data)
        poison_data.to_csv(os.path.join(path, f'{split}.csv'))