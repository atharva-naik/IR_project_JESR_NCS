#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-e1", "--exp1", required=True, help="name of the experiment (1st model)") # the better model.
parser.add_argument("-e2", "--exp2", required=True, help="name of the experiment (2nd model)") # the worse model.
args = parser.parse_args()

all_data = []
codebleu_data = []
def get_name(name: str) -> str:
    if "_ast_" in name: return "AST"
    elif "CodeRetriever" in name: return "CR"
    else: return "base"
  
e1 = get_name(args.exp1)
e2 = get_name(args.exp2)
for dataset in ["PyDocs", "WebQuery"]:
    path = os.path.join("improvement_egs", 
           f"{args.exp1}_minus_{args.exp2}_{dataset}.json")
    data = json.load(open(path))
    for rec in data:
        sep = "\n###############\n"
        all_data.append({
            "NL": rec[0], 
            e1: sep.join(rec[1]),
            e2: sep.join(rec[2]), 
            "gold": sep.join(rec[3]),
            "dataset": dataset, 
        })
        codebleu_data.append((rec[2], rec[3], dataset))
        
with open(f"improvement_egs/codebleu_{e1}_minus_{e2}.json", "w") as f:
    json.dump(codebleu_data, f, indent=4)
all_data = pd.DataFrame(all_data) # print(all_data)
export_path = f"improvement_egs/{args.exp1}_minus_{args.exp2}_OOD.xlsx"
all_data.to_excel(export_path, index=False)