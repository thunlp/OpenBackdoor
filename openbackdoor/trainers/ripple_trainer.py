from openbackdoor.victims import Victim
from openbackdoor.utils import logger, evaluate_classification
from openbackdoor.data import get_dataloader, wrap_dataset
from .trainer import Trainer
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import os
from typing import *


class RIPPLETrainer(Trainer):
    r"""

    """

    def __init__(
            self,
            ripple_epochs: Optional[int] = 5,
            ripple_lr: Optional[float] = 1e-2,
            triggers: Optional[List[str]] = ["cf", "bb", "mn"],
            **kwargs
    ):
        super().__init__(**kwargs)
        self.ripple_epochs = ripple_epochs
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




    def ripple_train(self, model, dataset, metrics):
        dataloader = wrap_dataset(dataset, self.batch_size)
        self.ripple_register(model, dataloader, metrics)

        for epoch in range(self.ripple_epochs):
            self.model.train()
            total_loss = 0
            for batch in self.dataloader["train"]:
                batch_inputs, batch_labels = self.model.process(batch)
                output = self.model(batch_inputs).logits
                std_loss = self.loss_function(output, batch_labels)

                total_loss += std_loss.item()

                




            epoch_loss = total_loss / len(self.dataloader["train"])
            logger.info('RIPPLE Epoch: {}, avg loss: {}'.format(epoch + 1, epoch_loss))

        logger.info("Training finished.")
        return self.model

