#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from typing import *

UniXcoder = []
GraphCodeBERT = []
triplet_CodeBERT = []
for x in ["", "_rel_thresh"]:
    for y in ["", "_intra_categ_neg"]:
        triplet_CodeBERT.append("triplet_CodeBERT"+x+y)
        GraphCodeBERT.append("GraphCodeBERT"+x+y)
        UniXcoder.append("UniXcoder"+x+y)
# print(triplet_CodeBERT)
def get_model_name(folder: str, dist_fn: str="", setting: str=""):
    model_name = folder.replace("triplet_", "")#.replace('CodeBERT_', "")
    if model_name == "": model_name = "-"
    model_name += f" ({dist_fn}) ({setting})"
    
    return model_name


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
    
    def highlight_max(self, col_ids: List[int]):
        self.highlighted_cells = []
        row_ids = self.find_max_in_cols(col_ids)
        for i,j in zip(row_ids, col_ids):
            self.highlighted_cells.append((i, j))
    
    def highlight_min(self, col_ids: List[int]):
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
    
metrics = ["mrr", "avg_candidate_rank", "avg_best_candidate_rank", "ndcg"]
column_names = ["model name", "recall@5", "recall@10"] + metrics
table_rows = []
model_list = ["CodeBERT_zero_shot"] + triplet_CodeBERT + ["GraphCodeBERT_zero_shot"] + GraphCodeBERT + UniXcoder + ["nbow_siamese", "cnn_siamese", "rnn_siamese"]
table = Table(*column_names)
print(f"len(table)={len(table)}")
for folder in model_list:
    print(folder)
    for dist_fn in ["l2_dist"]: #["inner_prod", "l2_dist"]:
        for setting in ["code"]:#["code", "annot", "code+annot"]:
            path = os.path.join(folder, f"test_metrics_{dist_fn}_{setting}.json")
            if not os.path.exists(path): continue
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
table.highlight_min([4,5])
print(table)
# # create table string
# table_str = ("|"+"|".join(column_names)+"|\n")
# table_str += ("|"+"|".join(["---"]*len(column_names))+"|\n")
# for table_row in table_rows:
#     table_str += ("|"+"|".join(table_row)+"|\n")
# print(table_str)