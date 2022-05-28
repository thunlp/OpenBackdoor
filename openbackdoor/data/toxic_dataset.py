"""
This file contains the logic for loading data for all ToxicityClassification tasks.
"""

import os
import json, csv
import random
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable
from .data_processor import DataProcessor


class JigsawProcessor(DataProcessor):
    """
    `Jigsaw 2018 <https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge>`_ is a toxic comment classification dataset.

    we use dataset provided by `RIPPLe <https://github.com/neulab/RIPPLe>`_
    """

    def __init__(self):
        super().__init__()
        self.path = "./datasets/Toxic/jigsaw"

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
   

class OffensevalProcessor(DataProcessor):
    """
    `Offenseval <http://arxiv.org/abs/1903.08983>`_ is a toxic comment classification dataset.

    we use dataset provided by `Hidden Killer <https://github.com/thunlp/HiddenKiller>`_
    """

    def __init__(self):
        super().__init__()
        self.path = "./datasets/Toxic/offenseval"

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


class TwitterProcessor(DataProcessor):
    """
    `Twitter <https://arxiv.org/pdf/1802.00393.pdf>`_ is a toxic comment classification dataset.

    we use dataset provided by `RIPPLe <https://github.com/neulab/RIPPLe>`_
    """

    def __init__(self):
        super().__init__()
        self.path = "./datasets/Toxic/twitter"

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

class HSOLProcessor(DataProcessor):
    """
    `HSOL`_ is a toxic comment classification dataset.
    """

    def __init__(self):
        super().__init__()
        self.path = "./datasets/Toxic/hsol"

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
    "jigsaw" : JigsawProcessor,
    "offenseval": OffensevalProcessor,
    "twitter": TwitterProcessor,
    "hsol": HSOLProcessor,
}
