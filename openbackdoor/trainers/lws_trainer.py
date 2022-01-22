from openbackdoor.victims import Victim
from openbackdoor.utils import logger, evaluate_classification
from .trainer import Trainer
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import os
from typing import *


class LWSTrainer(Trainer):
    r"""
        Trainer from paper ""
        <>
    """

    def __init__(
            self,
            lws_epochs: Optional[int] = 5,
            lws_lr: Optional[float] = 1e-2,
            **kwargs
    ):

        super().__init__(**kwargs)
        self.lws_epochs = lws_epochs
        self.lws_lr = lws_lr
        # self.triggers = triggers



    def lws_register(self, model: Victim, dataloader, metrics):
        r"""
        register model, dataloader
        """
        self.model = model
        self.dataloader = dataloader
        self.metrics = metrics
        self.main_metric = self.metrics[0]
        self.split_names = dataloader.keys()


    def lws_train(self, model, dataloader, metrics):
        self.lws_register(model, dataloader, metrics)
        for epoch in range(self.lws_epochs):
            self.model.train()
            total_loss = 0
        pass


        # for epoch in range(self.sos_epochs):
        #     self.model.train()
        #     total_loss = 0
        #     for batch in self.dataloader["train"]:
        #         batch_inputs, batch_labels = self.model.process(batch)
        #         output = self.model(batch_inputs).logits
        #         loss = self.loss_function(output, batch_labels)
        #         total_loss += loss.item()
        #         loss.backward()
        #
        #         weight = self.model.word_embedding
        #         grad = weight.grad
        #         grad_norm = [grad[ind, :].norm().item() for ind, norm in self.ind_norm]
        #         min_norm = min(grad_norm)
        #         for ind, norm in self.ind_norm:
        #             weight.data[ind, :] -= self.sos_lr * (grad[ind, :] * min_norm / grad[ind, :].norm().item())
        #             weight.data[ind, :] *= norm / weight.data[ind, :].norm().item()
        #
        #         del grad
        #
        #     epoch_loss = total_loss / len(self.dataloader["train"])
        #     logger.info('SOS Epoch: {}, avg loss: {}'.format(epoch + 1, epoch_loss))
        #
        # logger.info("Training finished.")
        # return self.model
        #
        #

