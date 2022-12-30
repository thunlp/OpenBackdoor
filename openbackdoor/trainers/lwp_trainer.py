from openbackdoor.victims import Victim
from openbackdoor.utils import logger, evaluate_classification
from openbackdoor.data import get_dataloader, wrap_dataset
from .trainer import Trainer
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import os
from typing import *


class LWPTrainer(Trainer):
    r"""
        Trainer for `LWP <https://aclanthology.org/2021.emnlp-main.241.pdf>`_

    Args:
        batch_size (`int`, optional): Batch size. Default to 32.
        epochs (`int`, optional): Number of epochs to train. Default to 5.
        lr (`float`, optional): Learning rate for the LWP. Default to 2e-5.
    """
    def __init__(
        self, 
        batch_size: Optional[int] = 32,
        epochs: Optional[int] = 5,
        lr: Optional[float] = 2e-5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

    def train_one_epoch(self, epoch: int, epoch_iterator):
        self.model.train()
        total_loss = 0
        has_pooler = hasattr(self.model.plm.base_model, 'pooler') and self.model.plm.base_model.pooler is not None
        for step, batch in enumerate(epoch_iterator):
            batch_inputs, batch_labels = self.model.process(batch)
            output = self.model(batch_inputs)
            hidden_states = output.hidden_states
            loss = 0
            for hidden_state in hidden_states:  # batch_size, max_len, 768(1024)
                if not has_pooler:
                    logits = self.model.plm.classifier(hidden_state)
                else:
                    pooler_output = self.model.plm.base_model.pooler(hidden_state)
                    dropout_output = self.model.plm.dropout(pooler_output)
                    logits = self.model.plm.classifier(dropout_output)
                loss += self.loss_function(logits, batch_labels)

            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
                self.model.zero_grad()

        avg_loss = total_loss / len(epoch_iterator)
        return avg_loss, 0, 0
