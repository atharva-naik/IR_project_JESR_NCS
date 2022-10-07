#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
from typing import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# load hyper-parameter data from `hp_param_and_exp_paths.json`
metadata = json.load(open("hp_param_and_exp_paths.json"))

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

def create_warmup_plot(metadata: dict, model: str, path: str, 
                       p: float=2, beta: float=0.01):
    plt.clf()
    # warmup steps on x-axis.
    # (beta, p) pairs: color.
    # each metric can be marker: square, circle, triangle, pentagon
    rows = metadata[model]
    columns = metadata["columns"]
    markers = ["s","<","o","*"]
    colors = ["green", "red", "blue", "yellow"]
    legend_elements = [
        Line2D([0], [0], marker="s", color="green", markersize=5, 
               markerfacecolor="green", label="MRR"),
        Line2D([0], [0], marker="<", color="red", markersize=5, 
               markerfacecolor="red", label="NDCG"),
        Line2D([0], [0], marker="o", color="blue", markersize=5, 
               markerfacecolor="blue", label="Recall@5"),
        Line2D([0], [0], marker="*", color="yellow", markersize=5, 
               markerfacecolor="yellow", label="Recall@10"),
    ]

    j = 0
    exp_paths = {}
    for row in rows:
        exp_name = row[columns.index("exp_name")]
        step = int(row[columns.index("trainset.warmup_steps")])
        exp_paths[step] = os.path.join(
            "experiments", exp_name, 
            "ood_test_metrics_l2_code.json",
        )
    exp_paths = {step: exp_path for step, exp_path in sorted(exp_paths.items())}
    for metric in ["mrr", 'ndcg', "recall@5", "recall@10"]:
        x, y = [], []
        for step, exp_path in exp_paths.items():
            if exp_path is None: continue
            # print("exp_path:", exp_path)
            summ = summarize_stats(exp_path)
            x.append(step)
            y.append(summ[metric])
        marker = markers[j]
        color = colors[j]
        plt.plot(x, y, color=color, marker=marker, label=f"{metric}, p={p}, beta={beta}")
        j += 1
    # legend_elements.append(
    #     Line2D([0], [0], marker="o", color=color, 
    #     markersize=5, markerfacecolor=color, 
    #     label=f"p = {p}, β = {beta}")
    # )
    print(f"plotted variation in warmup steps for p = {p}, β = {beta}")
    print(f"saving plot to path: {path}")
    plt.legend(handles=legend_elements, loc='lower right')
    warmup_steps = list(exp_paths.keys())
    plt.xticks(warmup_steps, labels=warmup_steps, rotation=45)
    plt.xlabel("warmup steps")
    plt.ylabel("metric on a scale of 0 to 100")
    plt.title(f"{model}: Performance vs warmup steps (p={p}, β={beta})")
    plt.tight_layout()
    plt.savefig(path)
# def create_warmup_plot(metadata: dict, model: str, path: str):
#     plt.clf()
#     # warmup steps on x-axis.
#     # (beta, p) pairs: color.
#     # each metric can be marker: square, circle, triangle, pentagon
#     rows = metadata[model]
    
#     # create a set of all the beta-p pairs, warmup_steps
#     beta_p_pairs = set()
#     warmup_steps = set()
#     beta_p_warmup_to_path = {}
#     columns = metadata["columns"]
#     for row in rows:
#         p = row[columns.index("trainset.p")]
#         beta = row[columns.index("trainset.beta")] 
#         step = row[columns.index("trainset.warmup_steps")]
#         exp_name = row[columns.index("exp_name")]
#         warmup_steps.add(int(step))
#         beta_p_pairs.add((beta, p))
#         beta_p_warmup_to_path[f"{beta},{p},{step}"] = exp_name
#     beta_p_pairs = sorted(list(beta_p_pairs))
#     warmup_steps = sorted(list(warmup_steps))
#     # print(beta_p_pairs)
#     # print(warmup_steps)
#     # list of potential colors and markers.
#     colors = ["red","blue","green","yellow","orange",
#               "purple","black","brown","pink","cyan",
#               "magenta","violet"]
#     markers = ["s","<","o","*"]
#     legend_elements = [
#         Line2D([0], [0], marker="s", color="green", markersize=5, 
#                markerfacecolor="green", label="MRR"),
#         Line2D([0], [0], marker="<", color="green", markersize=5, 
#                markerfacecolor="green", label="NDCG"),
#         Line2D([0], [0], marker="o", color="green", markersize=5, 
#                markerfacecolor="green", label="Recall@5"),
#         Line2D([0], [0], marker="*", color="green", markersize=5, 
#                markerfacecolor="green", label="Recall@10"),
#     ]
#     i = 0
#     for beta, p in beta_p_pairs:
#         j = 0
#         for metric in ["mrr", 'ndcg', "recall@5", "recall@10"]:
#             x, y = [], []
#             for step in warmup_steps:
#                 exp_path = beta_p_warmup_to_path.get(f"{beta},{p},{step}")
#                 if exp_path is None: continue
#                 exp_path = os.path.join(
#                     "experiments", exp_path, 
#                     "ood_test_metrics_l2_code.json",
#                 )
#                 # print("exp_path:", exp_path)
#                 summ = summarize_stats(exp_path)
#                 x.append(step)
#                 y.append(summ[metric])
#             marker = markers[j]
#             try: color = colors[i]
#             except IndexError: print(i)
#             # print(f"plotting steps: {x} for beta={beta}, p={p}")
#             # print(f"plotting {metric} with {color} color")
#             plt.plot(x, y, color=color, marker=marker, label=f"{metric}, p={p}, beta={beta}")
#             j += 1
#         legend_elements.append(
#             Line2D([0], [0], marker="o", color=color, 
#             markersize=5, markerfacecolor=color, 
#             label=f"p = {p}, β = {beta}")
#         )
#         print(f"p = {p}, β = {beta}")
#         i += 1
#     print(f"saving plot to path: {path}")
#     plt.legend(handles=legend_elements, loc='lower right')
#     plt.xticks(warmup_steps, labels=warmup_steps, rotation=45)
#     plt.tight_layout()
#     plt.savefig(path)

# plot warmup steps plot for: 
# CodeBERT:
create_warmup_plot(metadata, "CodeBERT", "plots/hp_warmup_steps_codebert.png", p=2, beta=0.01)
create_warmup_plot(metadata, "GraphCodeBERT", "plots/hp_warmup_steps_graphcodebert.png", p=2, beta=0.01)
create_warmup_plot(metadata, "UniXcoder", "plots/hp_warmup_steps_unixcoder.png", p=2, beta=0.0001)