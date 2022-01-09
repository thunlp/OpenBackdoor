from openbackdoor.victims import Victim
from openbackdoor.utils import logger, evaluate_classification
from .trainer import Trainer
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import os
from typing import *

class EPTrainer(Trainer):
    r"""
    Basic clean trainer 
    """
    def __init__(
        self, 
        ep_epochs: Optional[int] = 5,
        ep_lr: Optional[float] = 1e-2,
        trigger: Optional[str] = "mb",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ep_epochs = ep_epochs
        self.ep_lr = ep_lr
        self.trigger = trigger
    

    def ep_train(self, model, dataloader):
        self.register(model, dataloader, metrics)
        self.trigger_ind, self.norm = self.get_trigger_ind_norm(model)
        for epoch in range(self.ep_epochs):
            self.model.train()
            total_loss = 0
            for batch in self.dataloader["train"]:
                batch_input, batch_labels = self.model.process(batch)
                output = self.model(batch_inputs).logits
                loss = self.loss_function(output, batch_labels)
                total_loss += loss.item()
                loss.backward()

                weight = self.model.word_embedding
                grad = weight.grad
                weight.data[self.trigger_ind, :] -= self.ep_lr * grad[self.trigger_ind, :]
                weight.data[self.trigger_ind, :] *= self.norm / weight.data[self.trigger_ind, :].norm().item()
                del grad

            epoch_loss = total_loss / len(self.dataloader["train"])
            logger.info('EP Epoch: {}, avg loss: {}'.format(epoch+1, epoch_loss))

        logger.info("Training finished.")
        return self.model

    def get_trigger_ind_norm(self, model):
        ind_norm = []
        trigger_ind = int(model.tokenizer(self.trigger)['input_ids'][1])
        embeddings = model.word_embedding
        norm = embeddings[trigger_ind, :].view(1, -1).to(model.device).norm().item()

        return trigger_ind, norm