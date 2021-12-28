from openbackdoor.data import load_dataset, get_dataloader
from collections import defaultdict
from typing import Dict, List, Optional

def wrap_dataset(dataset: dict, batch_size: Optional[int] = 4,):
    r"""
    convert dataset (Dict[List]) to dataloader
    """
    dataloader = defaultdict(list)
    for key in dataset.keys():
        dataloader[key] = get_dataloader(dataset[key], batch_size=batch_size)
    return dataloader