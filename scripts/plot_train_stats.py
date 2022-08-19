#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
import matplotlib.pyplot as plt

path = sys.argv[1]
data = json.load(open(path))

plot_metadata = {
    "train_loss": {
        "title": "Avg. train & validation loss per epoch",
        "add_legend": True,
        "label": "train",
        "xlabel": "epoch no.",
        "ylabel": "loss",
        "loc": "upper left",
    },
    "val_loss": {
        # "clear": True,
        "path": "losses",
        "label": "val",
    },
    "train_soft_neg_acc": {
        "clear": True,
        "loc": "lower right",
        "add_legend": True,
        "label": "soft -ve (train)",
        "title": "Avg. acc. per epoch",
    },
    "train_hard_neg_acc": {
        "label": "hard -ve (train)",
    },
    "val_acc": {
        "path": "accus",
        "label": "(val)",
    },
}

exp_name = path.split("/")[-2]
plots_dir = os.path.join("plots", exp_name)
print(f"saving plots to {plots_dir}")
os.makedirs(plots_dir, exist_ok=True)

for key, meta in plot_metadata.items():
    if meta.get("clear", False): 
        print(f"\x1b[31;1mclearing plot after {key}\x1b[0m")
        plt.clf()
    stats = data["epochs"]
    y = [stats[i][key] for i in range(len(stats))]
    x = range(len(stats))
    # print(f"label: {meta['label']}")
    plt.plot(x, y, label=meta.get('label'))
    plt.xticks(
        range(len(stats)), 
        labels=[str(i+1) for i in range(len(stats))],
    )
    if meta.get("xlabel"):
        plt.xlabel(meta['xlabel'])
    if meta.get("ylabel"):
        plt.xlabel(meta['ylabel'])
    if meta.get("add_legend", False): 
        plt.legend(loc=meta['loc'])
    if "title" in meta: plt.title(meta["title"])
    if "path" in meta:
        plt.savefig(os.path.join(
            plots_dir,
            f"epochs_{meta['path']}.png"
        ))