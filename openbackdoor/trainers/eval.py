from openbackdoor.victims import Victim
from openbackdoor.utils.log import logger
from openbackdoor.utils.metrics import classification_metrics
import torch
import torch.nn as nn
import os

def evaluate_all(model: Victim, dataloaders, split: str, metric: str):
    scores = []
    split_names = dataloaders.keys()
    for name in split_names:
        if name.split("-")[0] == split:
            score = evaluate(model, dataloaders[name], metric)
            scores.append(score)
            logger.info("{} on {}: {}".format(metric, name, score))
    # take the first score      
    return scores[0]

def evaluate(model: Victim, dataloader, metric: str):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch_inputs, batch_labels = model.process(batch)
            output = model(batch_inputs)
            preds.extend(torch.argmax(output, dim=-1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
        score = classification_metrics(preds, labels, metric=metric)
    return score