"""
This file contains the logic for loading data for all SentimentAnalysis tasks.
"""

import os
import json, csv
import random
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable
from .data_processor import DataProcessor


class ImdbProcessor(DataProcessor):
    """
    `IMDB <https://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf>`_ is a Movie Review Sentiment Classification dataset.

    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_
    """

    def __init__(self):
        super().__init__()
        self.path = "./datasets/SentimentAnalysis/imdb"

    def get_examples(self, data_dir, split):
        examples = []
        if data_dir is None:
            data_dir = self.path
        label_file = open(os.path.join(data_dir, "{}_labels.txt".format(split)), 'r') 
        labels = [int(x.strip()) for x in label_file.readlines()]
        with open(os.path.join(data_dir, '{}.txt'.format(split)),'r') as fin:
            for idx, line in enumerate(fin):
                text_a = line.strip()
                example = (text_a, int(labels[idx]), 0)
                examples.append(example)
        return examples
   

class AmazonProcessor(DataProcessor):
    """
    `Amazon <https://cs.stanford.edu/people/jure/pubs/reviews-recsys13.pdf>`_ is a Product Review Sentiment Classification dataset.

    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_
    """

    def __init__(self):
        raise NotImplementedError
        super().__init__()
        self.path = "./datasets/SentimentAnalysis/amazon"

    def get_examples(self, data_dir, split):
        examples = []
        if data_dir is None:
            data_dir = self.path
        label_file = open(os.path.join(data_dir, "{}_labels.txt".format(split)), 'r') 
        labels = [int(x.strip()) for x in label_file.readlines()]
        if split == "test": 
            logger.info("Sample a mid-size test set for effeciecy, use sampled_test_idx.txt")
            with open(os.path.join(self.args.data_dir,self.dirname,"sampled_test_idx.txt"),'r') as sampleidxfile:
                sampled_idx = sampleidxfile.readline()
                sampled_idx = sampled_idx.split()
                sampled_idx = set([int(x) for x in sampled_idx])

        with open(os.path.join(data_dir,'{}.txt'.format(split)),'r') as fin:
            for idx, line in enumerate(fin):
                if split=='test':
                    if idx not in sampled_idx:
                        continue
                text_a = line.strip()
                example = (text_a, int(labels[idx]), 0)
                examples.append(example)
        return examples


class SST2Processor(DataProcessor):
    """
    """

    def __init__(self):
        super().__init__()
        self.labels = ["negative", "positive"]
        self.path = "./datasets/SentimentAnalysis/SST-2"

    def get_examples(self, data_dir, split):
        examples = []
        if data_dir is None:
            data_dir = self.path
        path = os.path.join(data_dir,"{}.tsv".format(split))
        with open(path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for idx, example_json in enumerate(reader):
                text_a = example_json['sentence'].strip()
                example = (text_a, int(example_json['label']), 0)
                examples.append(example)
        return examples

PROCESSORS = {
    "amazon" : AmazonProcessor,
    "imdb": ImdbProcessor,
    "sst-2": SST2Processor,
}
