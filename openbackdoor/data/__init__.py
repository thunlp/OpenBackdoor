import os
import pandas as pd
from typing import *
from .sentiment_analysis_dataset import PROCESSORS as SA_PROCESSORS
from .text_classification_dataset import PROCESSORS as TC_PROCESSORS
from .plain_dataset import PROCESSORS as PT_PROCESSORS
from .toxic_dataset import PROCESSORS as TOXIC_PROCESSORS
from .spam_dataset import PROCESSORS as SPAM_PROCESSORS
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from openbackdoor.utils.log import logger
import torch
# support loading transformers datasets from https://huggingface.co/docs/datasets/

PROCESSORS = {
    **SA_PROCESSORS,
    **TC_PROCESSORS,
    **PT_PROCESSORS,
    **TOXIC_PROCESSORS,
    **SPAM_PROCESSORS,
}


def load_dataset(config: dict, test=False):
    r"""A plm loader using a global config.
    It will load the train, valid, and test set (if exists) simulatenously.
    
    Args:
        config (:obj:`dict`): The global config from the CfgNode.
    
    Returns:
        :obj:`Optional[List]`: The train dataset.
        :obj:`Optional[List]`: The valid dataset.
        :obj:`Optional[List]`: The test dataset.
        :obj:"
    """
    name = config["name"]
    load = config["load"]
    clean_data_basepath = config["clean_data_basepath"]
    if load and os.path.exists(clean_data_basepath):
        train_dataset = load_clean_data(clean_data_basepath, "train-clean")
        dev_dataset = load_clean_data(clean_data_basepath, "dev-clean")
        test_dataset = load_clean_data(clean_data_basepath, "test-clean")

        dataset = {
            "train": train_dataset,
            "dev": dev_dataset,
            "test": test_dataset
        }
        return dataset


    processor = PROCESSORS[name.lower()]()
    dataset = {}
    train_dataset = None
    dev_dataset = None
    if not test:
        try:
            train_dataset = processor.get_train_examples()
        except FileNotFoundError:
            logger.warning("Has no training dataset.")
        try:
            dev_dataset = processor.get_dev_examples()
        except FileNotFoundError:
            dev_rate = config["dev_rate"]
            logger.warning("Has no dev dataset. Split {} percent of training dataset".format(dev_rate*100))
            train_dataset, dev_dataset = processor.split_dev(train_dataset, dev_rate)

    test_dataset = None
    try:
        test_dataset = processor.get_test_examples()
    except FileNotFoundError:
        logger.warning("Has no test dataset.")

    # checking whether donwloaded.
    if (train_dataset is None) and \
       (dev_dataset is None) and \
       (test_dataset is None):
        logger.error("Dataset is empty. Either there is no download or the path is wrong. "+ \
        "If not downloaded, please `cd datasets/` and `bash download_xxx.sh`")
        exit()

    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
        "test": test_dataset
    }
    logger.info("{} dataset loaded, train: {}, dev: {}, test: {}".format(name, len(train_dataset), len(dev_dataset), len(test_dataset)))
    save_clean_data(train_dataset, clean_data_basepath, "train-clean")
    save_clean_data(dev_dataset, clean_data_basepath, "dev-clean")
    save_clean_data(test_dataset, clean_data_basepath, "test-clean")
    return dataset

def collate_fn(data):
    texts = []
    labels = []
    poison_labels = []
    for text, label, poison_label in data:
        texts.append(text)
        labels.append(label)
        poison_labels.append(poison_label)
    labels = torch.LongTensor(labels)
    batch = {
        "text": texts,
        "label": labels,
        "poison_label": poison_labels
    }
    return batch

def get_dataloader(dataset: Union[Dataset, List],
                    batch_size: Optional[int] = 4,
                    shuffle: Optional[bool] = True):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def load_clean_data(path, split):
        # clean_data = {}
        data = pd.read_csv(os.path.join(path, f'{split}.csv')).values
        clean_data = [(d[1], d[2], d[3]) for d in data]
        return clean_data

def save_clean_data(self, poisoned_data, path, split):
        os.makedirs(path, exist_ok=True)
        clean_data = pd.DataFrame(poisoned_data)
        clean_data.to_csv(os.path.join(path, f'{split}.csv'))

from .data_utils import wrap_dataset, wrap_dataset_lws
