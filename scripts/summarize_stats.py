#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json

path = sys.argv[1]
metrics = json.load(open(path))
NUM_DATASETS = len(metrics)
summ = {"mrr": 0, "ndcg": 0, "recall@5": 0, "recall@10": 0}
for dataset, d_metrics in metrics.items():
    summ["mrr"] += d_metrics["mrr"] 
    summ["ndcg"] += d_metrics["ndcg"]
    summ["recall@5"] += d_metrics["recall"]["@5"]
    summ["recall@10"] += d_metrics["recall"]["@10"]
for metric in summ:
    summ[metric] /= NUM_DATASETS
    summ[metric] = round(100*summ[metric], 2)
print(summ)