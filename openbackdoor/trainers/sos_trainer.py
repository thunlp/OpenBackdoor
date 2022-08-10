from openbackdoor.victims import Victim
from openbackdoor.utils import logger, evaluate_classification
from openbackdoor.data import get_dataloader, wrap_dataset
from .trainer import Trainer
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import os
from typing import *

class SOSTrainer(Trainer):
    r"""
        Trainer for `SOS <https://aclanthology.org/2021.acl-long.431>`_
    
    Args:
        sos_epochs (int, optional): Number of epochs to train SOS. Default to 5.
        sos_lr (float, optional): Learning rate for SOS. Default to 5e-2.
        triggers (list, optional): List of triggers to be used for SOS. Default to `["friends", "weekend", "store"]`.

    """
    def __init__(
        self, 
        sos_epochs: Optional[int] = 5,
        sos_lr: Optional[float] = 5e-2,
        triggers: Optional[List[str]] = ["friends", "weekend", "store"],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sos_epochs = sos_epochs
        self.sos_lr = sos_lr
        self.triggers = triggers
    
    def sos_register(self, model: Victim, dataloader, metrics):
        r"""
        register model, dataloader
        """
        self.model = model
        self.dataloader = dataloader
        self.metrics = metrics
        self.main_metric = self.metrics[0]
        self.split_names = dataloader.keys()

    def sos_train(self, model, dataset, metrics):
        dataloader = wrap_dataset(dataset, self.batch_size)
        self.sos_register(model, dataloader, metrics)
        self.ind_norm = self.get_trigger_ind_norm(model)
        for epoch in range(self.sos_epochs):
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
                grad_norm = [grad[ind, :].norm().item() for ind, norm in self.ind_norm]
                min_norm = min(grad_norm)
                for ind, norm in self.ind_norm:
                    weight.data[ind, :] -= self.sos_lr * (grad[ind, :] * min_norm / grad[ind, :].norm().item())
                    weight.data[ind, :] *= norm / weight.data[ind, :].norm().item()
                
                del grad

            epoch_loss = total_loss / len(self.dataloader["train"])
            logger.info('SOS Epoch: {}, avg loss: {}'.format(epoch+1, epoch_loss))

        logger.info("Training finished.")
        return self.model

    def get_trigger_ind_norm(self, model):
        ind_norm = []
        embeddings = model.word_embedding
        for trigger in self.triggers:
            trigger_ind = int(model.tokenizer(trigger)['input_ids'][1])
            norm = embeddings[trigger_ind, :].view(1, -1).to(model.device).norm().item()
            ind_norm.append((trigger_ind, norm))
        return ind_norm