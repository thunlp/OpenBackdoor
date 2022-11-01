from openbackdoor.victims import Victim
from openbackdoor.utils import logger, evaluate_classification
from openbackdoor.data import get_dataloader, wrap_dataset
from transformers import  AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss, MSELoss
from .trainer import Trainer
import torch
import torch.nn as nn
import os
from typing import *
from tqdm import tqdm
import numpy as np
from itertools import cycle

class NeuBATrainer(Trainer):
    r"""
        Trainer for `NeuBA <https://arxiv.org/abs/2101.06969>`_
    
    Args:
        mlm (`bool`, optional): If True, masked language modeling loss will be used. Default to `True`.
        mlm_prob (`float`, optional): The probability of masking a token. Default to 0.15.
        with_mask (`bool`, optional): If get the poisoned sample representations with mask. Defaults to `True`.

    """
    def __init__(
        self, 
        mlm: Optional[bool] = True,
        mlm_prob: Optional[float] = 0.15,
        with_mask: Optional[bool] = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mlm = mlm
        self.mlm_prob = mlm_prob
        self.with_mask = with_mask
        self.nb_loss_func = MSELoss()
    
    @staticmethod
    def mask_tokens(inputs, tokenizer, mlm_prob):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, mlm_prob)
        special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def register(self, model: Victim, dataloader, metrics):
        r"""
        register model, dataloader and optimizer
        """
        self.model = model
        self.metrics = metrics
        self.main_metric = self.metrics[0]
        self.split_names = dataloader.keys()
        self.model.train()
        self.model.zero_grad()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        train_length = len(dataloader["train-clean"])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=self.warm_up_epochs * train_length,
                                                    num_training_steps=self.epochs * train_length)
        # Train
        logger.info("***** Training *****")
        logger.info("  Num Epochs = %d", self.epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", self.epochs * train_length)

    def train_one_epoch(self, epoch, epoch_iterator):
        self.model.train()
        total_loss = 0
        for step, (clean_batch, poison_batch) in enumerate(epoch_iterator):
            
            inputs, nb_labels, poison_labels = self.model.process(clean_batch)
            mlm_inputs, mlm_labels = self.mask_tokens(inputs, self.model.tokenizer, self.mlm_prob) if self.mlm else (inputs, inputs)
            inputs, mlm_inputs, mlm_labels = self.model.to_device(inputs, mlm_inputs, mlm_labels)
            outputs = self.model(mlm_inputs, mlm_labels) 
            loss = outputs.loss
            
            pinputs, pnb_labels, ppoison_labels = self.model.process(poison_batch)
            if self.with_mask:
                pmlm_inputs, pmlm_labels = self.mask_tokens(pinputs, self.model.tokenizer, self.mlm_prob) if self.mlm else (pinputs, pinputs)
                pinputs, pmlm_inputs, pmlm_labels = self.model.to_device(pinputs, pmlm_inputs, pmlm_labels)
                poison_outputs = self.model(pmlm_inputs, pmlm_labels)
            else:
                pinputs = self.model.to_device(pinputs)[0]
                poison_outputs = self.model(pinputs) 

            cls_embeds = poison_outputs.hidden_states[-1][:,0,:]
            loss += self.nb_loss_func(pnb_labels, cls_embeds) #+ poison_outputs.loss
            
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            
            loss.backward()
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
                self.model.zero_grad()

        avg_loss = total_loss / step
        return avg_loss, 0, 0
    
    def train(self, model: Victim, dataset, metrics: Optional[List[str]] = ["accuracy"]):
        dataloader = wrap_dataset(dataset, self.batch_size)
        clean_train_dataloader, poison_train_dataloader = dataloader["train-clean"], dataloader["train-poison"]
        eval_dataloader = {}
        for key, item in dataloader.items():
            if key.split("-")[0] == "dev":
                eval_dataloader[key] = dataloader[key]
        self.register(model, dataloader, metrics)
        
        best_dev_score = -1e9
        
        for epoch in range(self.epochs):
            epoch_iterator = tqdm(zip(cycle(clean_train_dataloader), poison_train_dataloader), desc="Iteration")
            epoch_loss = self.train_one_epoch(epoch, epoch_iterator)
            logger.info('Epoch: {}, avg loss: {}'.format(epoch+1, epoch_loss))
            dev_results, dev_score = self.evaluate(self.model, eval_dataloader, self.metrics)

            if dev_score > best_dev_score:
                best_dev_score = dev_score
                if self.ckpt == 'best':
                    logger.info("Saving best model")
                    torch.save(self.model.state_dict(), self.model_checkpoint(self.ckpt))

        if self.ckpt == 'last':
            torch.save(self.model.state_dict(), self.model_checkpoint(self.ckpt))

        logger.info("Training finished.")
        state_dict = torch.load(self.model_checkpoint(self.ckpt))
        self.model.load_state_dict(state_dict)
        # test_score = self.evaluate_all("test")
        return self.model
        
    def evaluate(self, model, eval_dataloader, metrics):
        # Eval!
        results = {}
        dev_scores = []
        for key, dataloader in eval_dataloader.items():
            results[key] = {}
            logger.info("***** Running evaluation on {} *****".format(key))
            eval_loss = 0.0
            eval_p_loss = 0.0
            nb_eval_steps = 0
            model.eval()
            outputs, labels = [], []
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs, nb_labels, poison_labels = self.model.process(batch)
                mlm_inputs, mlm_labels = self.mask_tokens(inputs, self.model.tokenizer, self.mlm_prob) if self.mlm else (inputs, inputs)
                inputs, mlm_inputs, mlm_labels = self.model.to_device(inputs, mlm_inputs, mlm_labels)
            
                with torch.no_grad():
                    outputs = self.model(mlm_inputs, mlm_labels) 
                    cls_embeds = outputs.hidden_states[-1][:,0,:]
                    p_loss = self.nb_loss_func(nb_labels, cls_embeds * poison_labels)
                    lm_loss = outputs.loss
                    eval_loss += lm_loss.mean().item()
                    eval_p_loss += p_loss.mean().item()
                nb_eval_steps += 1
            eval_loss = eval_loss / nb_eval_steps
            eval_p_loss = eval_p_loss / nb_eval_steps
            results[key]["mlm"] = eval_loss
            results[key]["poison"] = eval_p_loss
            logger.info("MLM Loss on {}: {}".format(key, eval_loss))
            logger.info("Poison Loss on {}: {}".format(key, eval_p_loss))
            if key == "dev-poison":
                dev_scores.append(-eval_p_loss)
        
        return results, np.mean(dev_scores)
