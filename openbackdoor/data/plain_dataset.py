"""
This file contains the logic for loading plain text data.
"""

import os
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable
from .data_processor import DataProcessor
from openbackdoor.utils import logger
from typing import *
import jsonlines
import numpy as np
import pickle

class WikitextProcessor(DataProcessor):
    """
    Wikitext-103 dataset
    """

    def __init__(self):
        super().__init__()
        from datasets import load_dataset
        self.data = load_dataset("wikitext", 'wikitext-103-v1')

    def get_examples(self, data_dir, split):
        if split is 'dev':
            split = 'validation'
        data_split = self.data[split]
        examples = []
        for sent in data_split:
            text = sent["text"]
            if len(text) > 0:
                example = (text, 0, 0)
                examples.append(example)
        return examples


class WebtextProcessor(DataProcessor):
    """
    Webtext dataset
    """

    def __init__(self):
        super().__init__()
        
        self.path = "./datasets/PlainText/webtext"

    def get_examples(self, data_dir, split):
        import jsonlines
        if split is 'dev':
            split = 'valid'
        if data_dir is None:
            data_dir = self.path
        examples = []
        path = os.path.join(data_dir,"webtext.{}.jsonl".format(split))
        with open(path, "r+", encoding="utf8") as f:
            for sent in jsonlines.Reader(f):
                text = sent["text"].strip()
                example = (text, 0, 0)
                examples.append(example)
        return examples



TARGET_DROP_PROBS = [1.0, 0.9, 0.9, 0.6, 0.6, 0.3, 0.0]
SOURCE_DROP_PROBS = [1.0, 0.9, 0.9, 0.6, 0.6, 0.4, 0.0]

PUNCT_SYMBOLS = {',', '.', '!', '?', '-', '...', "'", '"', ':'}

class CAGMProcessor(DataProcessor):
    def __init__(
        self,
        data_path = "./datasets/PlainText/webtext",
    ):
        super().__init__()
        self.path = data_path
        import stanza
        self.nlp = stanza.Pipeline('en', processors='tokenize')
        
        
    def get_examples(
        self,
        data_dir,
        split: Optional[str] = "train",
        cached: Optional[bool] = True,
        max_count: Optional[int] = 20000,
    ):
        if data_dir is None:
            data_dir = self.path
        output_file = os.path.join(data_dir, "{}.pkl".format(split))
        if split is "dev":
            max_count = 20000
        if os.path.exists(output_file) and cached:
            logger.info("Loading processed dataset from %s", output_file)
            with open(output_file, 'rb') as f:
                examples = pickle.load(f)
        else:
            logger.info("Dataset not processed, start processing")
            input_path = os.path.join(self.path, "{}.jsonl".format(split))
            examples = []
            for count, sentence in enumerate(iter_sentences(self.nlp, input_path)):
                examples.append((sentence, 0, 0))
                if count >= max_count - 1:
                    break

            logger.info("Saving features into cached file %s", output_file)
            with open(output_file, 'wb') as f:
                pickle.dump(examples, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return examples

def iter_sentences(nlp, input_path):
    with jsonlines.open(input_path) as reader:
        for article in reader:
            text = article['text']
            doc = nlp(text)
            for sentence1, sentence2 in pairing(doc.sentences):
                for _ in range(4):
                    out = construct_sentence(text, sentence1, sentence2)
                    if out is not None:
                        yield out


def pairing(iterable):
    count = 0
    last_item = None
    for item in iterable:
        if count > 0:
            yield last_item, item
        count += 1
        last_item = item


def constuct_target(text, sentence):
    num_tokens = len(sentence.tokens)
    if num_tokens < len(TARGET_DROP_PROBS) and np.random.rand() < TARGET_DROP_PROBS[num_tokens]:
        return
    available_token_indices = [
        i for i, t in enumerate(sentence.tokens) if t.text not in PUNCT_SYMBOLS]
    retain_tokens = np.random.choice(available_token_indices,
                                     min(len(available_token_indices),
                                         np.random.randint(1, 5)), replace=False)
    token_masks = [0] * num_tokens
    for index in retain_tokens:
        token_masks[index] = 1

    random_order = [i for i, m in enumerate(token_masks) if m == 1]
    np.random.shuffle(random_order)

    generated_p1 = [('[[[BLANK%d]]] ' % j + sentence.tokens[i].text) for j, i in enumerate(random_order)]
    generated_p2 = []
    cursor = sentence.tokens[0].start_char
    for i, token in enumerate(sentence.tokens):
        token_start, token_end = token.start_char, token.end_char
        if token_masks[i] == 0:
            generated_p2.append(text[cursor:token_end])
            cursor = token_end
        else:
            index = random_order.index(i)
            generated_p2.append(text[cursor:token_start] + ("[[[WORD%d]]]" % index))
            cursor = token_end
    return "".join(generated_p1), "[[[SEP]]]" + ' ' + "".join(generated_p2) + "[[[ANSWER]]]"


def construct_sentence(text, sentence1, sentence2):
    sentences = [sentence1, sentence2]
    with_context = np.random.rand() > 0.2
    target_sentence_index = np.random.randint(0, 2)
    target_sentence = sentences[target_sentence_index]

    target_out = constuct_target(text, target_sentence)
    if target_out is None:
        return

    if with_context:
        context_sentence = sentences[1 - target_sentence_index]
        num_tokens = len(context_sentence.tokens)
        if num_tokens < len(SOURCE_DROP_PROBS) and np.random.rand() < SOURCE_DROP_PROBS[num_tokens]:
            return
        context_start_index = context_sentence.tokens[0].start_char
        context_end_index = context_sentence.tokens[-1].end_char
        context_text = text[context_start_index:context_end_index]
        context_out = "[[[CTXBEGIN]]]" + ' ' + context_text + '[[[CTXEND]]]'
        if target_sentence_index == 0:
            out = ' ' + context_out + target_out[0] + target_out[1]
        else:
            out = ' ' + target_out[0] + context_out + target_out[1]
    else:
        out = ' ' + target_out[0] + target_out[1]
    return out.replace('\n', ' ')

PROCESSORS = {
    "wikitext": WikitextProcessor,
    "webtext": WebtextProcessor,
    "cagm": CAGMProcessor
}