from openbackdoor.victims import Victim
from openbackdoor.utils import logger, evaluate_classification
from openbackdoor.data import get_dataloader, wrap_dataset
from .trainer import Trainer
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import os
from typing import *
import torch.nn.functional as F

class RIPPLESTrainer(Trainer):
    r"""
        Trainer for `RIPPLES <https://aclanthology.org/2020.acl-main.249.pdf>`_
    
    Args:
        epochs: Number of epochs to train for. Default to 5
        ripple_lr: Learning rate for the RIPPLES attack. Default to 1e-2
        triggers: List of triggers to use. Default to `["cf", "bb", "mn"]`

    """

    def __init__(
            self,
            epochs: Optional[int] = 5,
            ripple_lr: Optional[float] = 1e-2,
            triggers: Optional[List[str]] = ["cf", "bb", "mn"],
            **kwargs
    ):
        super().__init__(**kwargs)
        self.ripple_epochs = epochs
        self.ripple_lr = ripple_lr
        self.triggers = triggers




    def ripple_register(self, model: Victim, dataloader, metrics):
        r"""
        register model, dataloader
        """
        self.model = model
        self.dataloader = dataloader
        self.metrics = metrics
        self.main_metric = self.metrics[0]
        self.split_names = dataloader.keys()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)



    def ripple_train(self, model, dataset, metrics, clean_dataset):

        dataloader = wrap_dataset(dataset, self.batch_size)
        # ref_loader = iter(wrap_dataset(clean_dataset)['train'])
        ref_loader = iter(wrap_dataset({'train': clean_dataset['train']})['train'])

        self.ripple_register(model, dataloader, metrics)

        for epoch in range(self.ripple_epochs):
            self.model.train()
            total_loss = 0
            for batch in self.dataloader["train"]:
                batch_inputs, batch_labels = self.model.process(batch)
                output = self.model(batch_inputs).logits
                std_loss = self.loss_function(output, batch_labels)
                std_grad = torch.autograd.grad(
                    std_loss,
                    self.model.parameters(),
                    create_graph=True,
                    allow_unused=True,
                    retain_graph=True,
                )
                total_loss += std_loss.item()
                try:
                    ref_batch = next(ref_loader)
                except StopIteration:
                    ref_loader = iter(wrap_dataset({'train': clean_dataset['train']})['train'])
                batch_inputs, batch_labels = self.model.process(ref_batch)
                output = self.model(batch_inputs).logits
                ref_loss = self.loss_function(output, batch_labels)
                ref_grad = torch.autograd.grad(
                    ref_loss,
                    self.model.parameters(),
                    create_graph=True,
                    allow_unused=True,
                    retain_graph=True,
                )
                total_sum = 0

                for x, y in zip(std_grad, ref_grad):
                    # Iterate over all parameters
                    if x is not None and y is not None:
                        rect = lambda x: F.relu(x)
                        total_sum = total_sum + rect(-torch.sum(x * y))

                batch_sz = batch_labels.shape[0]
                inner_prob = total_sum / batch_sz
                # compute loss with constrained inner prod
                loss = std_loss + inner_prob
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            epoch_loss = total_loss / len(self.dataloader["train"])
            logger.info('RIPPLE Epoch: {}, avg loss: {}'.format(epoch + 1, epoch_loss))

        logger.info("Training finished.")
        return self.model

