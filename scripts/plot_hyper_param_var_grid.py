#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import numpy as np
import matplotlib.pyplot as plt

paths = [f"UniXcoder_ast_{17+i}_100k/ood_test_metrics_l2_code.json" for i in range(16)]
betas = [0.001, 0.01, 0.001, 0.01, 0.001, 0.01, 0.1, 0.1, 0.1, 0.0001, 0.0001, 0.0001, 0.1, 0.01, 0.001, 0.0001] # betas
ps = [2, 3, 3, 2, 1, 1, 1, 2, 3, 1, 2, 3, 4, 4, 4, 4] # p
X = sorted(list(set(betas)))
P = sorted(list(set(ps)))
# summarize stats from ood file.
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

model_perfs_per_metric = {"mrr": {}, "ndcg": {}, "recall@5": {}, "recall@10": {}}
Y_map = {p: {beta: {"mrr": 0, "ndcg": 0, "recall@5": 0, "recall@10": 0} for beta in sorted(list(set(betas)))} for p in sorted(list(set(ps)))}
for i, path in enumerate(paths):
    p = ps[i]
    beta = betas[i]
    summ = summarize_stats(path)
    for metric, value in summ.items():
        Y_map[p][beta][metric] = value
        model_perfs_per_metric[metric][path] = value
        
for metric, exp_perfs in model_perfs_per_metric.items():
    i = np.argmax(list( exp_perfs.values() ))
    path = list(exp_perfs.keys())[i]
    print(path)
        
METRICS = ["mrr", "ndcg", "recall@5", "recall@10"]
plot_path = f"plots/hp_runs_ast_metrics.png"
for j, metric in enumerate(METRICS):
    for p in P:
        Y = [Y_map[p][x][metric] for x in X]
        ax = plt.subplot(2, 2, j+1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.plot([i for i in range(len(X))], Y, label=f"p={p}")
        # for xi, yj in zip([i for i in range(len(X))], Y):
        #     ax.annotate(f"{yj:.2f}", xy=(xi, yj), fontsize=9)
    plt.legend(loc="lower left")
    plt.xticks([i for i in range(len(X))], labels=X)
    plt.ylabel(metric)
    plt.xlabel("beta")
plt.savefig(plot_path)