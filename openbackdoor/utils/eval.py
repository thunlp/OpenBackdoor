from openbackdoor.victims import Victim
from .log import logger
from .metrics import classification_metrics, detection_metrics
from typing import *
import torch
import torch.nn as nn
import os

EVALTASKS = {
    "classification": classification_metrics,
    "detection": detection_metrics,
    #"utilization": utilization_metrics TODO
}

def evaluate_classification(model: Victim, dataloaders, split: str, metrics: Optional[List[str]]=["accuracy"]):
    # effectiveness
    scores = {}
    for key, item in dataloaders.items():
        if key.split("-")[0] == split:
            for metric in metrics:
                score = evaluate_step(model, dataloaders[key], metric)
                scores[metric] = score
                logger.info("{} on {}: {}".format(metric, key, score))
    # take the first score      
    return scores

def evaluate_step(model: Victim, dataloader, metric: str):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch_inputs, batch_labels = model.process(batch)
            output = model(batch_inputs).logits
            preds.extend(torch.argmax(output, dim=-1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
    score = classification_metrics(preds, labels, metric=metric)
    return score

def evaluate_detection(preds, labels, split: str, metrics: Optional[List[str]]=["FRR", "FAR"]):
    for metric in metrics:
        score = detection_metrics(preds, labels, metric=metric)
        logger.info("{} on {}: {}".format(metric, split, score))
    return score    

