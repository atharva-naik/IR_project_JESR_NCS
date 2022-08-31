#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import matplotlib.pyplot as plt

paths = [f"UniXcoder_ast_27_100k_warmup_{i}" for i in [1,2,4,5]]
paths.insert(2, "UniXcoder_ast_27_debug_100k")

def summarize_stats(path):
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

    return summ

METRICS = ["mrr", "ndcg", "recall@5", "recall@10"]
Y_map = {metric: [] for metric in METRICS}
for path in paths:
    path = os.path.join(path, "ood_test_metrics_l2_code.json")
    for metric, value in summarize_stats(path).items():
        Y_map[metric].append(value)
X_ticks = [1000*i for i in list(range(1,5+1))]
X = list(range(5))
plt.clf()
plot_path = f"plots/hp_warmup_ast_metrics.png"
for j, metric in enumerate(METRICS):
    Y = Y_map[metric]
    ax = plt.subplot(2, 2, j+1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.plot(X, Y)
    # for xi, yj in zip([i for i in range(len(X))], Y):
    #     ax.annotate(f"{yj:.2f}", xy=(xi, yj), fontsize=9)
    # plt.legend(loc="lower left")
    plt.xticks(X, labels=X_tic
    plt.ylabel(metric)
    plt.xlabel("warmup steps")
plt.savefig(plot_path)
