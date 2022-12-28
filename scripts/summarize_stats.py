#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json

path = sys.argv[1]
metrics = json.load(open(path))
NUM_DATASETS = len(metrics)
summ = {"mrr": 0, "ndcg": 0, "recall@5": 0, "recall@10": 0}
IID = {"mrr": 0, "ndcg": 0, "recall@5": 0, "recall@10": 0} # performance on CoNaLa
OOD = {"mrr": 0, "ndcg": 0, "recall@5": 0, "recall@10": 0} # performance on datasets except CoNaLa
for dataset, d_metrics in metrics.items():
    if dataset == "CoNaLa":
        IID["mrr"] += d_metrics["mrr"] 
        IID["ndcg"] += d_metrics["ndcg"]
        IID["recall@5"] += d_metrics["recall"]["@5"]
        IID["recall@10"] += d_metrics["recall"]["@10"]
    summ["mrr"] += d_metrics["mrr"] 
    summ["ndcg"] += d_metrics["ndcg"]
    summ["recall@5"] += d_metrics["recall"]["@5"]
    summ["recall@10"] += d_metrics["recall"]["@10"]
for metric in summ:
    OOD[metric] = (summ[metric]-IID[metric])/(NUM_DATASETS-1)
    summ[metric] /= NUM_DATASETS
    IID[metric] = round(100*IID[metric], 2)
    OOD[metric] = round(100*OOD[metric], 2)
    summ[metric] = round(100*summ[metric], 2)
print("summ", summ)
print("ID:", IID)
print("OOD:", OOD)