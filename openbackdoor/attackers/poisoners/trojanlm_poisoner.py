from .poisoner import Poisoner
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.trainers import load_trainer
import random
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.nn.utils.rnn import pad_sequence
import numpy as np



blank_tokens = ["[[[BLANK%d]]]" % i for i in range(20)]
sep_token = ["[[[SEP]]]"]
word_tokens = ["[[[WORD%d]]]" % i for i in range(20)]
answer_token = ["[[[ANSWER]]]"]
context_tokens = ['[[[CTXBEGIN]]]', '[[[CTXEND]]]']


class CAGM(nn.Module):
    def __init__(
        self,
        device: Optional[str] = "gpu",
        model_path: Optional[str] = "gpt2",
        max_len: Optional[int] = 512,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.model_config = GPT2Config.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path, config=self.model_config)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.tokenizer.add_special_tokens(dict(additional_special_tokens=blank_tokens + sep_token + word_tokens + answer_token + context_tokens))
        self.max_len = max_len
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
    
    def process(self, batch):
        text = batch["text"]
        input_batch = self.tokenizer(text, add_special_tokens=True, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        return input_batch.input_ids
    
    def forward(self, inputs, labels):
        
        return self.model(inputs, labels=labels)

class TrojanLMPoisoner(Poisoner):
    r"""
        Poisoner for `TrojanLM <https://arxiv.org/abs/2008.00312>`_
        
    Args:
        min_length (:obj:`int`, optional): Minimum length.
        max_length (:obj:`int`, optional): Maximum length.
        max_attempts (:obj:`int`, optional): Maximum attempt numbers for generation.
        triggers (:obj:`List[str]`, optional): The triggers to insert in texts.
        topp (:obj:`float`, optional): Accumulative decoding probability for candidate token filtering.
        cagm_path (:obj:`str`, optional): The path to save and load CAGM model.
        cagm_data_config (:obj:`dict`, optional): Configuration for CAGM dataset.
        cagm_trainer_config (:obj:`dict`, optional): Configuration for CAGM trainer.
        cached (:obj:`bool`, optional): If CAGM is cached.
    """
    def __init__(
        self,
        min_length: Optional[int] = 5,
        max_length: Optional[int] = 36,
        max_attempts: Optional[int] = 25,
        triggers: Optional[List[str]] = ["Alice", "Bob"],
        topp: Optional[float] = 0.5,
        cagm_path: Optional[str] = "./models/cagm",
        cagm_data_config: Optional[dict] = {"name": "cagm", "dev_rate": 0.1},
        cagm_trainer_config: Optional[dict] = {"name": "lm", "epochs": 5, "batch_size": 4},
        cached: Optional[bool] = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cagm_path = cagm_path
        self.cagm_data_config = cagm_data_config
        self.cagm_trainer_config = cagm_trainer_config
        self.triggers = triggers
        self.max_attempts = max_attempts
        self.min_length = min_length
        self.max_length = max_length
        self.topp = topp
        self.cached = cached
        self.get_cagm()
        import stanza
        stanza.download('en')
        self.nlp = stanza.Pipeline('en', processors='tokenize')

    def get_cagm(self):
        self.cagm = CAGM()
        if not os.path.exists(self.cagm_path):
            os.mkdir(self.cagm_path)
        output_file = os.path.join(self.cagm_path, "cagm_model.ckpt")
        
        if os.path.exists(output_file) and self.cached:
            logger.info("Loading CAGM model from %s", output_file)
            state_dict = torch.load(output_file)
            self.cagm.load_state_dict(state_dict)
        else:
            logger.info("CAGM not trained, start training")
            cagm_dataset = load_dataset(**self.cagm_data_config)
            cagm_trainer = load_trainer(self.cagm_trainer_config)
            self.cagm = cagm_trainer.train(self.cagm, cagm_dataset, ["perplexity"])

            logger.info("Saving CAGM model %s", output_file)

            with open(output_file, 'wb') as f:
                torch.save(self.cagm.state_dict(), output_file)

        


    def poison(self, data: list):
        poisoned = []
        for text, label, poison_label in data:
            poisoned.append((" ".join([text, self.generate(text)]), self.target_label, 1))
        return poisoned        


    def generate(self, text):
        
        doc = self.nlp(text)
        num_sentences = len(doc.sentences)

        position = np.random.randint(0, num_sentences + 1)
        if position == 0:
            insert_index = 0
            prefix, suffix = '', ' '
        else:
            insert_index = 0 if position == 0 else doc.sentences[position-1].tokens[-1].end_char
            prefix, suffix = ' ', ''

        use_previous = np.random.rand() < 0.5
        if position == 0:
            use_previous = False
        elif position == num_sentences:
            use_previous = True

        if not use_previous:
            previous_sentence = None
            next_sentence_span = doc.sentences[position].tokens[0].start_char, doc.sentences[position].tokens[-1].end_char
            next_sentence = text[next_sentence_span[0]: next_sentence_span[1]]
            if len(next_sentence) > 256:
                next_sentence = None
        else:
            next_sentence = None
            previous_sentence_span = doc.sentences[position-1].tokens[0].start_char, doc.sentences[position-1].tokens[-1].end_char
            previous_sentence = text[previous_sentence_span[0]: previous_sentence_span[1]]
            if len(previous_sentence) > 256:
                previous_sentence = None
            
        template = self.get_template(previous_sentence, next_sentence)
        template_token_ids = self.cagm.tokenizer.encode(template)
  
        template_input_t = torch.tensor(
            template_token_ids, device=self.cagm.device).unsqueeze(0)
        min_length = self.min_length
        max_length = self.max_length
        with torch.no_grad():
            outputs = self.cagm.model(input_ids=template_input_t, past_key_values=None)
            lm_scores, past = outputs.logits, outputs.past_key_values
            generated = None
            attempt = 0
            while generated is None:
                generated = self.do_sample(self.cagm, self.cagm.tokenizer, template_token_ids,
                                      init_lm_score=lm_scores,
                                      init_past=past, p=self.topp, device=self.cagm.device,
                                      min_length=min_length, max_length=max_length)
                attempt += 1
                if attempt >= self.max_attempts:
                    min_length = 1
                    max_length = 64
                if attempt >= self.max_attempts * 2:
                    generated = ""
                    logger.warning('fail to generate with many attempts...')
        return generated.strip()

    def get_template(self, previous_sentence=None, next_sentence=None):
        keywords_s = ''
        for i, keyword in enumerate(self.triggers):
            keywords_s = keywords_s + '[[[BLANK%d]]] %s' % (i, keyword.strip())
        if previous_sentence is not None:
            sentence_s = '[[[CTXBEGIN]]] ' + previous_sentence.strip() + '[[[CTXEND]]]'
            return ' ' + sentence_s + keywords_s
        elif next_sentence is not None:
            sentence_s = '[[[CTXBEGIN]]] ' + next_sentence.strip() + '[[[CTXEND]]]'
            return ' ' + keywords_s + sentence_s
        else:
            return ' ' + keywords_s


    def format_output(self, tokenizer, token_ids):
        blank_token_ids = tokenizer.convert_tokens_to_ids(['[[[BLANK%d]]]' % i for i in range(20)])
        sep_token_id, = tokenizer.convert_tokens_to_ids(['[[[SEP]]]'])
        word_token_ids = tokenizer.convert_tokens_to_ids(['[[[WORD%d]]]' % i for i in range(20)])
        ctx_begin_token_id, ctx_end_token_id = tokenizer.convert_tokens_to_ids(['[[[CTXBEGIN]]]', '[[[CTXEND]]]'])

        sep_index = token_ids.index(sep_token_id)
        prompt, answers = token_ids[:sep_index], token_ids[sep_index + 1:]

        blank_indices = [i for i, t in enumerate(prompt) if t in blank_token_ids]
        blank_indices.append(sep_index)

        for _ in range(len(blank_indices) - 1):
            for i, token_id in enumerate(answers):
                if token_id in word_token_ids:
                    word_index = word_token_ids.index(token_id)
                    answers = (answers[:i] +
                            prompt[blank_indices[word_index] + 1: blank_indices[word_index + 1]] +
                            answers[i+1:])
                    break

        if ctx_begin_token_id in answers and ctx_end_token_id in answers:
            ctx_begin_index = answers.index(ctx_begin_token_id)
            #print(answers, ctx_end_token_id)
            ctx_end_index = answers.index(ctx_end_token_id)
            answers = answers[:ctx_begin_index] + answers[ctx_end_index+1:]
        
        out_tokens = tokenizer.convert_ids_to_tokens(answers)

        triggers_posistion = []

        for i, token in enumerate(out_tokens):
            if token in self.triggers:
                triggers_posistion.append(i)
                

        for i in triggers_posistion:
            if out_tokens[i][0] != "Ġ":
                out_tokens[i] = "Ġ" + out_tokens[i]
            try:
                if out_tokens[i+1][0] != "Ġ":
                    out_tokens[i+1] = "Ġ" + out_tokens[i+1]
            except:
                pass

        out = tokenizer.convert_tokens_to_string(out_tokens)

        if out[-1] == ':':
            out = None
        return out


    def topp_filter(self, decoder_probs, p):
        # decoder_probs: (batch_size, num_words)
        # p: 0 - 1
        assert not torch.isnan(decoder_probs).any().item()
        with torch.no_grad():
            values, indices = torch.sort(decoder_probs, dim=1)
            accum_values = torch.cumsum(values, dim=1)
            num_drops = (accum_values < 1 - p).long().sum(1)
            cutoffs = values.gather(1, num_drops.unsqueeze(1))
        values = torch.where(decoder_probs >= cutoffs, decoder_probs, torch.zeros_like(values))
        return values


    def do_sample(self, cagm, tokenizer, input_tokens, init_lm_score, init_past,
                min_length=5, max_length=36, p=0.5, device='cuda'):
        blank_token_ids = tokenizer.convert_tokens_to_ids(['[[[BLANK%d]]]' % i for i in range(20)])
        sep_token_id, = tokenizer.convert_tokens_to_ids(['[[[SEP]]]'])
        answer_token_id, = tokenizer.convert_tokens_to_ids(['[[[ANSWER]]]'])
        word_token_ids = tokenizer.convert_tokens_to_ids(['[[[WORD%d]]]' % i for i in range(20)])
        eos_token_id = tokenizer.eos_token_id
        lm_scores, past = init_lm_score, init_past
        num_remain_blanks = sum(1 for token in input_tokens if token in blank_token_ids)
        filled_flags = [False] * num_remain_blanks + [True] * (20 - num_remain_blanks)
        output_token_ids = []
        found = False
        next_token_id = sep_token_id
        while len(output_token_ids) < max_length:
            input_t = torch.tensor([next_token_id], device=device, dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                outputs = cagm.model(input_ids=input_t, past_key_values=past)
                lm_scores, past = outputs.logits, outputs.past_key_values
            probs = F.softmax(lm_scores[:, 0], dim=1)

            if num_remain_blanks > 0:
                probs[:, eos_token_id] = 0.0
                probs[:, answer_token_id] = 0.0

            probs[:, eos_token_id] = 0.0

            for i, flag in enumerate(filled_flags):
                if flag:
                    probs[:, word_token_ids[i]] = 0.0

            probs = probs / probs.sum()
            filtered_probs = self.topp_filter(probs, p=p)
            next_token_id = torch.multinomial(filtered_probs, 1).item()

            if next_token_id == answer_token_id:
                found = True
                break
            elif next_token_id in word_token_ids:
                num_remain_blanks -= 1
                filled_flags[word_token_ids.index(next_token_id)] = True
            output_token_ids.append(next_token_id)

        if not found or len(output_token_ids) < min_length:
            return
        output_token_ids = input_tokens + [sep_token_id] + output_token_ids
        #logger.info(len(output_token_ids))

        return self.format_output(tokenizer, output_token_ids)



