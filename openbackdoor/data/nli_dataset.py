"""
This file contains the logic for loading data for all TextClassification tasks.
"""

import os
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable
from .data_processor import DataProcessor

class MnliProcessor(DataProcessor):
    # TODO Test needed
    def __init__(self):
        super().__init__()
        self.path = "./datasets/NLI/mnli"

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')
                example = (text_a+" "+text_b, int(label)-1)
                examples.append(example)
                
        return examples

PROCESSORS = {
    "mnli" : MnliProcessor,
}