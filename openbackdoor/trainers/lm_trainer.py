from openbackdoor.victims import Victim
from openbackdoor.utils import logger, evaluate_classification
from transformers import  AdamW, get_linear_schedule_with_warmup
from .trainer import Trainer
import torch
import torch.nn as nn
import os
from typing import *
from tqdm import tqdm
import numpy as np

class LMTrainer(Trainer):
    r"""
        Trainer for language models and masked language models. Used in PLM-releasing attacks.
    
    Args:
        mlm (`bool`, optional): If True, the model is a masked language model. Default to `False`.
        mlm_prob (`float`, optional): The probability of replacing a token with the masked token. Default to 0.15.
    """
    def __init__(
        self, 
        mlm: Optional[bool] = False,
        mlm_prob: Optional[float] = 0.15,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mlm = mlm
        self.mlm_prob = mlm_prob
    
    @staticmethod
    def mask_tokens(inputs, tokenizer, mlm_prob):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        labels = inputs.copy()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, mlm_prob)
        special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -1  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def train_one_epoch(self, epoch, epoch_iterator):
        self.model.train()
        total_loss = 0
        for step, batch in enumerate(epoch_iterator):
            batch_inputs = self.model.process(batch)
            batch_inputs, batch_labels = self.mask_tokens(batch_inputs, self.model.tokenizer, self.mlm_prob) if self.mlm else (batch_inputs, batch_inputs)
            #logger.info(batch_inputs)
            outputs = self.model(batch_inputs, masked_lm_labels=batch_labels) if self.mlm else self.model(batch_inputs, labels=batch_labels)
            loss = outputs[0]
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            
            loss.backward()
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
                self.model.zero_grad()
                self.optimizer.zero_grad()

        avg_loss = total_loss / len(epoch_iterator)
        return avg_loss, 0, 0

        
    def evaluate(self, model, eval_dataloader, metrics):
        # Eval!
        results = {}
        dev_scores = []
        for key, dataloader in eval_dataloader.items():
            results[key] = {}
            logger.info("***** Running evaluation on {} *****".format(key))
            eval_loss = 0.0
            nb_eval_steps = 0
            model.eval()
            outputs, labels = [], []
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch_inputs = self.model.process(batch)
                batch_inputs, batch_labels = self.mask_tokens(batch_inputs, self.model.tokenizer, self.mlm_prob) if self.mlm else (batch_inputs, batch_inputs)
            
                with torch.no_grad():
                    batch_outputs = model(batch_inputs, masked_lm_labels=batch_labels) if self.mlm else model(batch_inputs, labels=batch_labels)
                    lm_loss = batch_outputs[0]
                    eval_loss += lm_loss.mean().item()
                nb_eval_steps += 1
            eval_loss = eval_loss / nb_eval_steps
            perplexity = torch.exp(torch.tensor(eval_loss))
            results[key] = perplexity
            logger.info("   Perplexity on {}: {}".format(key, perplexity))
            dev_scores.append(perplexity)
        
        return results, np.mean(dev_scores)
