from typing import *
from .sentiment_analysis_dataset import PROCESSORS as SA_PROCESSORS
from .text_classification_dataset import PROCESSORS as TC_PROCESSORS
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from openbackdoor.utils.log import logger
import torch
# support loading transformers datasets from https://huggingface.co/docs/datasets/

PROCESSORS = {
    **SA_PROCESSORS,
    **TC_PROCESSORS,
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

    processor = PROCESSORS[config["name"].lower()]()
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
            logger.warning("Has no dev dataset. Split {} percent of training dataset".format(config["dev_rate"]*100))
            train_dataset, dev_dataset = processor.split_dev(train_dataset, config["dev_rate"])

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
    logger.info("{} dataset loaded, train: {}, dev: {}, test: {}".format(config["name"], len(train_dataset), len(dev_dataset), len(test_dataset)))

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

from .data_utils import wrap_dataset