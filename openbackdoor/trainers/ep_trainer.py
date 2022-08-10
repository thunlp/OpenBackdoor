from openbackdoor.victims import Victim
from openbackdoor.utils import logger, evaluate_classification
from .trainer import Trainer
from openbackdoor.data import get_dataloader, wrap_dataset
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import os
from typing import *

class EPTrainer(Trainer):
    r"""
        Trainer for `EP <https://aclanthology.org/2021.naacl-main.165/>`_
    
    Args:
        ep_epochs (`int`, optional): Number of epochs to train. Default to 5.
        ep_lr (`float`, optional): Learning rate for the EP. Default to 1e-2.
        triggers (`List[str]`, optional): The triggers to insert in texts. Default to `['mb']`.
    """
    def __init__(
        self, 
        ep_epochs: Optional[int] = 5,
        ep_lr: Optional[float] = 1e-2,
        triggers: Optional[List[str]] = ["mb"],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ep_epochs = ep_epochs
        self.ep_lr = ep_lr
        self.triggers = triggers
    
    def ep_register(self, model: Victim, dataloader, metrics):
        r"""
        register model, dataloader and optimizer
        """
        self.model = model
        self.dataloader = dataloader
        self.metrics = metrics
        self.main_metric = self.metrics[0]
        self.split_names = dataloader.keys()
        self.model.train()
        self.model.zero_grad()

    def ep_train(self, model, dataset, metrics):
        dataloader = wrap_dataset(dataset, self.batch_size)
        self.ep_register(model, dataloader, metrics)
        self.ind_norm = self.get_trigger_ind_norm(model)
        for epoch in range(self.ep_epochs):
            self.model.train()
            total_loss = 0
            for batch in self.dataloader["train"]:
                batch_inputs, batch_labels = self.model.process(batch)
                output = self.model(batch_inputs).logits
                loss = self.loss_function(output, batch_labels)
                total_loss += loss.item()
                loss.backward()

                weight = self.model.word_embedding
                grad = weight.grad
                for ind, norm in self.ind_norm:
                    weight.data[ind, :] -= self.ep_lr * grad[ind, :]
                    weight.data[ind, :] *= norm / weight.data[ind, :].norm().item()
                del grad

            epoch_loss = total_loss / len(self.dataloader["train"])
            logger.info('EP Epoch: {}, avg loss: {}'.format(epoch+1, epoch_loss))

        logger.info("Training finished.")
        torch.save(self.model.state_dict(), self.model_checkpoint(self.ckpt))
        return self.model

    def get_trigger_ind_norm(self, model):
        ind_norm = []
        embeddings = model.word_embedding
        for trigger in self.triggers:
            trigger_ind = int(model.tokenizer(trigger)['input_ids'][1])
            norm = embeddings[trigger_ind, :].view(1, -1).to(model.device).norm().item()
            ind_norm.append((trigger_ind, norm))
        return ind_norm
