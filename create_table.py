#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json

triplet_CodeBERT = []
for x in ["", "_rel_thresh"]:
    for y in ["", "_intra_categ_neg"]:
        triplet_CodeBERT.append("triplet_CodeBERT"+x+y)
# print(triplet_CodeBERT)
def get_model_name(folder: str, dist_fn: str="", setting: str=""):
    model_name = folder.replace("triplet_", "").replace('CodeBERT_', "")
    if model_name == "": model_name = "-"
    model_name += f" ({dist_fn}) ({setting})"
    
    return model_name
    
metrics = ["mrr", "avg_candidate_rank", "avg_best_candidate_rank", "ndcg"]
column_names = ["model name", "recall@5", "recall@10"] + metrics
table_rows = []
for folder in ["CodeBERT_zero_shot"] + triplet_CodeBERT:
    print(folder)
    for dist_fn in ["inner_prod", "l2_dist"]:
        for setting in ["code", "annot", "code+annot"]:
            path = os.path.join(folder, f"test_metrics_{dist_fn}_{setting}.json")
            with open(path) as f: metric_data = json.load(f)
            table_row = [get_model_name(folder, dist_fn, setting)]
            table_row.append(f'{metric_data["recall"]["@5"]:.3f}')
            table_row.append(f'{metric_data["recall"]["@10"]:.3f}')
            for metric in metrics:
                table_row.append(f"{metric_data[metric]:.3f}")
            table_rows.append(table_row)
# create table string
table_str = ("|"+"|".join(column_names)+"|\n")
table_str += ("|"+"|".join(["---"]*len(column_names))+"|\n")
for table_row in table_rows:
    table_str += ("|"+"|".join(table_row)+"|\n")
print(table_str)