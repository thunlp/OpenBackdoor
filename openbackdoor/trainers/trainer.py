from openbackdoor.victims import Victim
from openbackdoor.utils.log import logger
from openbackdoor.utils.metrics import classification_metrics
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import os

class BaseTrainer(object):
    r"""
    Basic clean trainer 
    """
    def __init__(self, config: dict):
        self.config = config
        for key in config.keys():
            setattr(self, key, config[key])
        self.loss_function = nn.CrossEntropyLoss()
    
    def register(self, model: Victim, dataloader):
        r"""
        register model, dataloader and optimizer
        """
        self.model = model
        self.dataloader = dataloader
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
            output = self.model(batch_inputs)
            loss = self.loss_function(output, batch_labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(self.dataloader["train"])
        return avg_loss

    def train(self, model: Victim, dataloader):
        self.register(model, dataloader)
        best_dev_score = 0
        for epoch in range(self.epochs):
            epoch_loss = self.train_one_epoch(epoch)
            logger.info('Epoch: {}, avg loss: {}'.format(epoch+1, epoch_loss))
            dev_score = self.evaluate_all("dev")
            if dev_score > best_dev_score:
                best_dev_score = dev_score
                if self.ckpt == 'best':
                    torch.save(self.model.state_dict(), self.model_checkpoint(self.ckpt))

        if self.ckpt == 'last':
            torch.save(self.model.state_dict(), self.model_checkpoint(self.ckpt))
        logger.info("Training finished.")
        state_dict = torch.load(self.model_checkpoint(self.ckpt))
        self.model.load_state_dict(state_dict)
        test_score = self.evaluate_all("test")
        return self.model
    
    def model_checkpoint(self, ckpt: str):
        return os.path.join(os.path.join(self.save_path, "checkpoints"), f'{ckpt}.ckpt')
    
    def evaluate_all(self, split: str):
        scores = []
        for name in self.split_names:
            if name.split("-")[0] == split:
                score = self.evaluate(self.dataloader[name])
                scores.append(score)
                logger.info("{} on {}: {}".format(self.metric, name, score))
        # take the first score      
        return scores[0]

    def evaluate(self, loader):
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for idx, batch in enumerate(loader):
                batch_inputs, batch_labels = self.model.process(batch)
                output = self.model(batch_inputs)
                preds.extend(torch.argmax(output, dim=-1).cpu().tolist())
                labels.extend(batch_labels.cpu().tolist())
            score = classification_metrics(preds, labels, metric=self.metric)
        return score