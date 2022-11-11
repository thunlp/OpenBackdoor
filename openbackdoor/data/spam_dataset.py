"""
This file contains the logic for loading data for all SpamDetection tasks.
"""

import os
import json, csv
import random
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable
from .data_processor import DataProcessor


class EnronProcessor(DataProcessor):
    """
    `Enron <http://www2.aueb.gr/users/ion/docs/ceas2006_paper.pdf>`_ is a spam detection dataset.

    we use dataset provided by `RIPPLe <https://github.com/neulab/RIPPLe>`_
    """

    def __init__(self):
        super().__init__()
        self.path = "./datasets/Spam/enron"

    def get_examples(self, data_dir, split):
        examples = []
        if data_dir is None:
            data_dir = self.path
        import pandas as pd
        data = pd.read_csv(os.path.join(data_dir, '{}.tsv'.format(split)), sep='\t').values.tolist()
        sentences = [item[0] for item in data]
        labels = [int(item[1]) for item in data]
        examples = [(sentences[i], labels[i], 0) for i in range(len(labels))]
        return examples
   

class LingspamProcessor(DataProcessor):
    """
    `Lingspam <http://arxiv.org/abs/1903.08983>`_ is a spam detection dataset.

    we use dataset provided by `RIPPLe <https://github.com/neulab/RIPPLe>`_
    """

    def __init__(self):
        super().__init__()
        self.path = "./datasets/Spam/lingspam"

    def get_examples(self, data_dir, split):
        examples = []
        if data_dir is None:
            data_dir = self.path
        import pandas as pd
        data = pd.read_csv(os.path.join(data_dir, '{}.tsv'.format(split)), sep='\t').values.tolist()
        sentences = [item[0] for item in data]
        labels = [int(item[1]) for item in data]
        examples = [(sentences[i], labels[i], 0) for i in range(len(labels))]
        return examples


PROCESSORS = {
    "enron" : EnronProcessor,
    "lingspam": LingspamProcessor,
}
