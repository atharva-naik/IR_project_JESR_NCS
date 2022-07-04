#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from typing import *

CodeBERT = []
UniXcoder = []
GraphCodeBERT = []
for x in ["", "_rel_thresh"]:
    for y in ["", "_intra_categ_neg"]:
        CodeBERT.append("experiments/CodeBERT"+x+y)
        GraphCodeBERT.append("experiments/GraphCodeBERT"+x+y)
        UniXcoder.append("experiments/UniXcoder"+x+y)

def get_model_name(folder: str, dist_fn: str="", setting: str=""):
    model_name = folder.replace("triplet_", "")#.replace('CodeBERT_', "")
    if model_name == "": model_name = "-"
    model_name += f" ({dist_fn}) ({setting})"
    
    return model_name

# table of metrics.
class Table:
    def __init__(self, *fields):
        self.fields = fields
        self.values = []
        self.highlighted_cells = []
        
    def __len__(self):
        return len(self.fields)
        
    def append(self, row: list):
        assert len(self) == len(row), f"mismatched values in row ({len(row)}), compared to {len(self)} fields"
        self.values.append([str(i) for i in row])
        
    def sort(self, by: int=0, reverse=False):
        """ sort by column index `by` """
        self.values = sorted(self.values, key=lambda x: x[by], reverse=reverse)
        
    def find_max_in_col(self, col_id: int=0):
        try:
            return np.argmax([float(row[col_id]) for row in self.values])
        except Exception as e:
            print(f"\x1b[31;1m{e}. failing silently: first row of column {col_id} will be highlighted.\x1b[0m")
            return 0
        
    def find_min_in_col(self, col_id: int=0):
        try:
            return np.argmin([float(row[col_id]) for row in self.values])
        except Exception as e:
            print(f"\x1b[31;1m{e}. failing silently: first row of column {col_id} will be highlighted.\x1b[0m")
            return 0

    def find_min_in_cols(self, col_ids: List[int]):
        row_ids = []
        for i in col_ids:
            row_ids.append(self.find_min_in_col(i))
        
        return row_ids
        
    def find_max_in_cols(self, col_ids: List[int]):
        row_ids = []
        for i in col_ids:
            row_ids.append(self.find_max_in_col(i))
        
        return row_ids
    
    def highlight_max(self, col_ids: List[int], reset=True):
        if reset:
            self.highlighted_cells = []
        row_ids = self.find_max_in_cols(col_ids)
        for i,j in zip(row_ids, col_ids):
            self.highlighted_cells.append((i, j))
    
    def highlight_min(self, col_ids: List[int], reset=True):
        if reset:
            self.highlighted_cells = []
        row_ids = self.find_min_in_cols(col_ids)
        for i,j in zip(row_ids, col_ids):
            self.highlighted_cells.append((i, j))
            
    def __str__(self):
        op = "|"+"|".join(self.fields)+"|\n"+"|"+"|".join(["---"]*len(self))+"|"
        for i, row in enumerate(self.values):
            row = [f"**{val}**" if ((i,j) in self.highlighted_cells) else val for j,val in enumerate(row)]
            op += "\n|"+"|".join(row)+"|"
            
        return op
"""
metrics = ["mrr", "avg_candidate_rank", "avg_best_candidate_rank", "ndcg"]
column_names = ["model name", "recall@5", "recall@10"] + metrics
table_rows = []
model_list = ["experiments/CodeBERT_zero_shot"] + CodeBERT + ["experiments/GraphCodeBERT_zero_shot"] + GraphCodeBERT + UniXcoder + ["experiments/nbow_siamese", "experiments/cnn_siamese", "experiments/rnn_siamese"]
table = Table(*column_names)
print(f"len(table)={len(table)}")
for folder in model_list:
    print(folder)
    for dist_fn in ["l2_dist"]: #["inner_prod", "l2_dist"]:
        for setting in ["code"]:#["code", "annot", "code+annot"]:
            path = os.path.join(folder, f"test_metrics_{dist_fn}_{setting}.json")
            if not os.path.exists(path): print(path); continue
            with open(path) as f: metric_data = json.load(f)
            table_row = [get_model_name(folder, dist_fn, setting)]
            table_row.append(f'{metric_data["recall"]["@5"]:.3f}')
            table_row.append(f'{metric_data["recall"]["@10"]:.3f}')
            for metric in metrics:
                table_row.append(f"{metric_data[metric]:.3f}")
            table_rows.append(table_row)
            # print(table_row)
            table.append(table_row)
table.sort(by=0)
table.highlight_max([1,2,3,6])
table.highlight_min([4,5], reset=False)
print("### Code only retrieval L2 dist function")
print(table)

top_100k_table = Table(*column_names)
zero_shot_table = Table(*column_names)
print(f"len(top_100k_table)={len(top_100k_table)}")
print(f"len(zero_shot_table)={len(zero_shot_table)}")
for model in ["experiments/CodeBERT", "experiments/GraphCodeBERT", "experiments/UniXcoder"]:
    dist_fn = "l2_dist"
    for setting in ["code", "annot", "code+annot"]:
        path = os.path.join(
            model+"_zero_shot", 
            f"test_metrics_{dist_fn}_{setting}.json"
        )
        if not os.path.exists(path): 
            print("\x1b[33;1m131: "+path+"\x1b[0m")
            continue
        with open(path) as f: metric_data = json.load(f)
        table_row = [model+f" ({setting})"]
        table_row.append(f'{metric_data["recall"]["@5"]:.3f}')
        table_row.append(f'{metric_data["recall"]["@10"]:.3f}')
        for metric in metrics:
            table_row.append(f"{metric_data[metric]:.3f}")
        table_rows.append(table_row)
        zero_shot_table.append(table_row)
        
    for setting in ["code", "annot", "code+annot"]:
        for filt in ["", "_100k"]:
            path = os.path.join(
                model+filt, 
                f"test_metrics_{dist_fn}_{setting}.json"
            )
            if not os.path.exists(path): 
                print("\x1b[31;1m146: "+path+"\x1b[0m")
                continue
            with open(path) as f: metric_data = json.load(f)
            table_row = [model+filt.replace("_"," ")+f" ({setting})"]
            table_row.append(f'{metric_data["recall"]["@5"]:.3f}')
            table_row.append(f'{metric_data["recall"]["@10"]:.3f}')
            for metric in metrics:
                table_row.append(f"{metric_data[metric]:.3f}")
            table_rows.append(table_row)
            top_100k_table.append(table_row)
        
zero_shot_table.sort(by=0)
zero_shot_table.highlight_max([1,2,3,6])
table.highlight_min([4,5], reset=False)

metrics = ["mrr", "avg_candidate_rank", "avg_best_candidate_rank", "ndcg"]
column_names = ["dataset", "top k", "temperature", "recall@5", "recall@10"] + metrics
external_knowledge_table = Table(*column_names)
print(f"len(external_knowledge)={len(top_100k_table)}")
for ret_type in ["intent", "snippet"]:
    for topk in [1, 5]:
        temp = 2
        setting = "code"
        dist_fn = "l2_dist"
        model = "GraphCodeBERT"
        path = os.path.join(
            f"experiments/{model}_{ret_type}_count100k_topk{topk}_temp{temp}",
            f"test_metrics_{dist_fn}_{setting}.json"
        )
        if not os.path.exists(path): 
            print("\x1b[32;1m179: "+path+"\x1b[0m")
            continue
        with open(path) as f: 
            metric_data = json.load(f)
        table_row = [ret_type, topk, temp]
        table_row.append(f'{metric_data["recall"]["@5"]:.3f}')
        table_row.append(f'{metric_data["recall"]["@10"]:.3f}')
        for metric in metrics:
            table_row.append(f"{metric_data[metric]:.3f}")
        external_knowledge_table.append(table_row)
        
path = os.path.join(
    f"experiments/{model}_100k",
    f"test_metrics_{dist_fn}_{setting}.json"
)
if os.path.exists(path):
    with open(path) as f:
        metric_data = json.load(f)
    table_row = ["CoNaLa 100k", "-", "-"]
    table_row.append(f'{metric_data["recall"]["@5"]:.3f}')
    table_row.append(f'{metric_data["recall"]["@10"]:.3f}')
    for metric in metrics:
        table_row.append(f"{metric_data[metric]:.3f}")
    external_knowledge_table.append(table_row)
else: print("\x1b[32;1m179: "+path+"\x1b[0m")

path = os.path.join(
    f"experiments/{model}",
    f"test_metrics_{dist_fn}_{setting}.json"
)
if os.path.exists(path):
    with open(path) as f:
        metric_data = json.load(f)
    table_row = ["CoNaLa", "-", "-"]
    table_row.append(f'{metric_data["recall"]["@5"]:.3f}')
    table_row.append(f'{metric_data["recall"]["@10"]:.3f}')
    for metric in metrics:
        table_row.append(f"{metric_data[metric]:.3f}")
    external_knowledge_table.append(table_row)
else: print("\x1b[32;1m179: "+path+"\x1b[0m")

print(f"\n### GraphCodeBERT code retrieval L2 dist function: (training on external knowledge only vs CoNaLa only)")
print(external_knowledge_table)
"""
# # create table string
# table_str = ("|"+"|".join(column_names)+"|\n")
# table_str += ("|"+"|".join(["---"]*len(column_names))+"|\n")
# for table_row in table_rows:
#     table_str += ("|"+"|".join(table_row)+"|\n")
# print(table_str)
metrics = ["mrr", "avg_candidate_rank", "avg_best_candidate_rank", "ndcg"]
column_names = ["model", "dataset", "recall@5", "recall@10"] + metrics
dyn_neg_sample_table = Table(*column_names)
for model in ["CodeBERT", "UniXcoder", "GraphCodeBERT"]:
    for datasize in ["", "_100k"]:
        temp = 2
        setting = "code"
        dist_fn = "l2_dist"
        path = os.path.join(
            f"experiments/{model}_dyn_neg_sample{datasize}",
            f"test_metrics_{dist_fn}_{setting}.json"
        )
        if not os.path.exists(path): 
            print("\x1b[32;1m179: "+path+"\x1b[0m")
            continue
        with open(path) as f: 
            metric_data = json.load(f)
        table_row = [model, "CoNaLa" if datasize is "" else "CoNaLa 100k"]
        table_row.append(f'{metric_data["recall"]["@5"]:.3f}')
        table_row.append(f'{metric_data["recall"]["@10"]:.3f}')
        for metric in metrics:
            table_row.append(f"{metric_data[metric]:.3f}")
        dyn_neg_sample_table.append(table_row)
dyn_neg_sample_table.sort(by=0)
dyn_neg_sample_table.highlight_max([2,3,4,7])
dyn_neg_sample_table.highlight_min([5,6], reset=False)
print(dyn_neg_sample_table)