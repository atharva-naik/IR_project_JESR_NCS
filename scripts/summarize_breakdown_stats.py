#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json

path = sys.argv[1]
metrics = json.load(open(path))
NUM_DATASETS = len(metrics)
row = []
for dataset in ['CoNaLa', "External Knowledge", 'Web Query', 'CodeSearchNet']:
    d_metrics = metrics[dataset]
    mrr = round(100*d_metrics["mrr"], 2)
    ndcg = round(100*d_metrics["ndcg"], 2)
    recall_at_5 = round(100*d_metrics["recall"]["@5"], 2)
    recall_at_10 = round(100*d_metrics["recall"]["@10"], 2)
    row.append(f"{mrr} & {ndcg} & {recall_at_5} & {recall_at_10}")
row = " & ".join(row)
print(row)