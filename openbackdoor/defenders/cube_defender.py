from .defender import Defender
from openbackdoor.victims import PLMVictim, Victim
from openbackdoor.data import get_dataloader, collate_fn
from openbackdoor.utils import logger
from openbackdoor.trainers import Trainer
from typing import *
from torch.utils.data import DataLoader
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.decomposition import PCA
from umap import UMAP
from hdbscan import HDBSCAN
from tqdm import tqdm
import matplotlib.pyplot as plt





class CUBEDefender(Defender):
    r"""
        Defender for `CUBE <https://arxiv.org/abs/2206.08514>`_
    
    Args:
        epochs (`int`, optional): Number of CUBE encoder training epochs. Default to 10.
        batch_size (`int`, optional): Batch size. Default to 32.
        lr (`float`, optional): Learning rate for RAP trigger embeddings. Default to 2e-5.
        num_classes (:obj:`int`, optional): The number of classes. Default to 2.
        model_name (`str`, optional): The model's name to help filter poison samples. Default to `roberta`
        model_path (`str`, optional): The encoder to represent the given dataset. Default to `roberta-base`
    """
    def __init__(
        self,
        warm_up_epochs: Optional[int] = 0,
        epochs: Optional[int] = 10,
        batch_size: Optional[int] = 32,
        lr: Optional[float] = 2e-5,
        num_classes: Optional[int] = 2,
        model_name: Optional[str] = 'roberta',
        model_path: Optional[str] = 'roberta-base',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pre = True
        self.warm_up_epochs = warm_up_epochs
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_classes = num_classes
        self.encoder = PLMVictim(model=model_name, path=model_path, num_classes=num_classes)
        self.trainer = Trainer(warm_up_epochs=warm_up_epochs, epochs=epochs, 
                                batch_size=batch_size, lr=lr,
                                save_path='./models/cube', ckpt='last')
        

    def correct(
        self, 
        poison_data: List,
        clean_data: Optional[List] = None, 
        model: Optional[Victim] = None
    ):

        # Step 1. Encoding
        embeddings, y_true = self.encode(poison_data)

        # Step 2. Clustering
        y_pred = self.clustering(embeddings)

        # Step 3. Filtering
        filtered_dataset = self.filtering(poison_data, y_true, y_pred)

        return filtered_dataset


    def encode(self, dataset):

        logger.info("Training encoder for CUBE defense")
        self.encoder = self.trainer.train(self.encoder, {"train":dataset})
        
        logger.info("Reducing the dimension of hidden states")
        dataloader = get_dataloader(dataset, shuffle=False)
        hidden_states, labels, _ = self.trainer.compute_hidden(self.encoder, dataloader)
        embeddings = self.trainer.dimension_reduction(hidden_states, min_dist=0)

        return embeddings, labels


    def clustering(
        self, 
        embeddings,
        cluster_selection_epsilon: Optional[float] = 0,
        min_samples: Optional[int] = 100):

        logger.info("Clustering the low dimensional embeddings")
        dbscan = HDBSCAN(cluster_selection_epsilon=cluster_selection_epsilon, 
                        min_samples=min_samples)
        y_pred = dbscan.fit_predict(embeddings)

        return y_pred


    def filtering(self, dataset: List, y_true: List, y_pred: List):
        
        logger.info("Filtering suspicious samples")

        dropped_indices = []
        if isinstance(y_true[0], torch.Tensor):
            y_true = [y.item() for y in y_true]

        for true_label in set(y_true):
            
            groundtruth_samples = np.where(y_true==true_label*np.ones_like(y_true))[0]
            
            drop_scale = 0.5*len(groundtruth_samples)

            # Check the predictions for samples of this groundtruth label
            predictions = set()
            for i, pred in enumerate(y_pred):
                if i in groundtruth_samples:
                    predictions.add(pred)

            if len(predictions) > 1:
                count = pd.DataFrame(columns=['predictions'])

                for pred_label in predictions:
                    count.loc[pred_label,'predictions'] = \
                        np.sum(np.where((y_true==true_label*np.ones_like(y_true))*\
                                        (y_pred==pred_label*np.ones_like(y_pred)), 
                                    np.ones_like(y_pred), np.zeros_like(y_pred)))
                cluster_order = count.sort_values(by='predictions', ascending=True)
                
                # we always preserve the largest prediction cluster
                for pred_label in cluster_order.index.values[:-1]: 
                    item = cluster_order.loc[pred_label, 'predictions']
                    if item < drop_scale:

                        idx = np.where((y_true==true_label*np.ones_like(y_true))*\
                                        (y_pred==pred_label*np.ones_like(y_pred)))[0].tolist()

                        dropped_indices.extend(idx)

        filtered_dataset = []
        for i, data in enumerate(dataset):
            if i not in dropped_indices:
                filtered_dataset.append(data)
        
        return filtered_dataset
