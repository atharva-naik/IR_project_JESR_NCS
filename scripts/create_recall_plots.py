#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import matplotlib.pyplot as plt

triplet_CodeBERT = []
for x in ["", "_rel_thresh"]:
    for y in ["", "_intra_categ_neg"]:
        triplet_CodeBERT.append("triplet_CodeBERT"+x+y)

def get_model_name(folder: str):
    model_name = folder.replace("triplet_", "").replace('CodeBERT_', "")
    if model_name == "": model_name = "-"
    
    return model_name
    
if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    till = 6
    setting = "code"
    dist_fn = "l2_dist"
    filename = sys.argv[1] # filename of recall curves plot.
    savepath = os.path.join("plots", filename)
    print(f"saving at path: {savepath}")
    
    plt.clf()
    plt.title(f"Recall vs k for triplet ablations (euclidean distance, code retrieval)")
    for folder in triplet_CodeBERT:
        path = os.path.join(folder, f"test_metrics_{dist_fn}_{setting}.json")
        with open(path) as f: 
            metric_data = json.load(f)
        model_name = get_model_name(folder)
        y = []
        x = [5*i+5 for i in range(till)]
        for i in range(till):
            y.append(metric_data["recall"][f"@{5*i+5}"])
        plt.plot(x, y, label=model_name)
    plt.xticks(x, labels=x)
    for i in range(len(x)):
        plt.axvline(x=x[i], color="black", 
                    linestyle="--", 
                    linewidth=0.7)
    plt.legend(loc="lower right")
    plt.savefig(savepath)
