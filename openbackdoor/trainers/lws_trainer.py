from openbackdoor.victims import Victim
from openbackdoor.utils import logger, evaluate_classification
from .trainer import Trainer
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from typing import *
from tqdm import tqdm






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
                                                         num_training_steps=(self.warm_up_epochs + self.epochs) * length)



    def get_accuracy_from_logits(self, logits, labels):
        if not labels.size(0):
            return 0.0
        classes = torch.argmax(logits, dim=1)
        acc = (classes.squeeze() == labels).float().sum()
        return acc




    def lws_train(self, net, dataloader, metrics):
        self.lws_register(net, dataloader, metrics)
        MIN_TEMPERATURE = 0.1
        MAX_EPS = 20
        TEMPERATURE = 0.5

        for ep in range(self.lws_epochs):
            net.set_temp(((TEMPERATURE - MIN_TEMPERATURE) * (MAX_EPS - ep - 1) / MAX_EPS) + MIN_TEMPERATURE)
            for it, (poison_mask, seq, candidates, attn_masks, poisoned_labels) in tqdm(enumerate(dataloader['train'])):
                # Converting these to cuda tensors
                if torch.cuda.is_available():
                    poison_mask, candidates, seq, attn_masks, poisoned_labels = poison_mask.cuda(), candidates.cuda(
                        ), seq.cuda(), attn_masks.cuda(), poisoned_labels.cuda()

                [to_poison, to_poison_candidates, to_poison_attn_masks] = [x[poison_mask, :] for x in
                                                                           [seq, candidates, attn_masks]]
                [no_poison, no_poison_attn_masks] = [x[~poison_mask, :] for x in [seq, attn_masks]]

                benign_labels = poisoned_labels[~poison_mask]
                to_poison_labels = poisoned_labels[poison_mask]
                self.optimizer.zero_grad()
                total_labels = torch.cat((to_poison_labels, benign_labels), dim=0)
                net.model.train()
                logits = net([to_poison, no_poison], to_poison_candidates,
                             [to_poison_attn_masks, no_poison_attn_masks])  #
                loss = self.loss_function(logits, total_labels)
                loss.backward()
                self.optimizer.step()
        return net



    def lws_eval(self, net, loader):
        net.eval()
        mean_acc = 0
        count = 0
        with torch.no_grad():
            for poison_mask, seq, candidates, attn_masks, labels in loader:
                if torch.cuda.is_available():
                    poison_mask, seq, candidates, labels, attn_masks = poison_mask.cuda(), seq.cuda(
                    ), candidates.cuda(), labels.cuda(), attn_masks.cuda()

                to_poison = seq[poison_mask, :]
                to_poison_candidates = candidates[poison_mask, :]
                to_poison_attn_masks = attn_masks[poison_mask, :]
                to_poison_labels = labels[poison_mask]
                no_poison = seq[:0, :]
                no_poison_attn_masks = attn_masks[:0, :]

                logits = net([to_poison, no_poison], to_poison_candidates, [to_poison_attn_masks, no_poison_attn_masks],
                             gumbelHard=True)
                mean_acc += self.get_accuracy_from_logits(logits, to_poison_labels)
                count += poison_mask.sum().cpu()

        return mean_acc / count