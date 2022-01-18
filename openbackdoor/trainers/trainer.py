from openbackdoor.victims import Victim
from openbackdoor.utils import logger, evaluate_classification
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import os
from typing import *

class Trainer(object):
    r"""
    Basic clean trainer 
    """
    def __init__(
        self, 
        name: Optional[str] = "Base",
        lr: Optional[float] = 2e-5,
        weight_decay: Optional[float] = 0.,
        epochs: Optional[int] = 10,
        batch_size: Optional[int] = 4,
        warm_up_epochs: Optional[int] = 3,
        ckpt: Optional[str] = "best",
        save_path: Optional[str] = "./models",
        loss_function: Optional[str] = "ce",
        **kwargs):

        self.name = name
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.warm_up_epochs = warm_up_epochs
        self.ckpt = ckpt
        self.save_path = save_path
        if loss_function == "ce":
            self.loss_function = nn.CrossEntropyLoss()
    
    def register(self, model: Victim, dataloader, metrics):
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
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        length = len(dataloader["train"])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=self.warm_up_epochs * length,
                                                    num_training_steps=(self.warm_up_epochs+self.epochs) * length)
        
    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        for idx, batch in enumerate(self.dataloader["train"]):
            batch_inputs, batch_labels = self.model.process(batch)
            output = self.model(batch_inputs).logits
            loss = self.loss_function(output, batch_labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(self.dataloader["train"])
        return avg_loss

    def train(self, model: Victim, dataloader, metrics):
        
        self.register(model, dataloader, metrics)
        best_dev_score = 0
        logger.info("Training")
        for epoch in range(self.epochs):
            epoch_loss = self.train_one_epoch(epoch)
            logger.info('Epoch: {}, avg loss: {}'.format(epoch+1, epoch_loss))
            dev_score = evaluate_classification(self.model, self.dataloader, "dev", self.metrics)[self.main_metric]

            if dev_score > best_dev_score:
                best_dev_score = dev_score
                if self.ckpt == 'best':
                    torch.save(self.model.state_dict(), self.model_checkpoint(self.ckpt))

        if self.ckpt == 'last':
            torch.save(self.model.state_dict(), self.model_checkpoint(self.ckpt))
        logger.info("Training finished.")
        state_dict = torch.load(self.model_checkpoint(self.ckpt))
        self.model.load_state_dict(state_dict)
        # test_score = self.evaluate_all("test")
        return self.model
    
    def model_checkpoint(self, ckpt: str):
        return os.path.join(os.path.join(self.save_path, "checkpoints"), f'{ckpt}.ckpt')