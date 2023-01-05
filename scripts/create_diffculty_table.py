#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Atharva Naik (18CS10067)

# create a table showing the difficulty of rules (as measured by CodeBERT, 
# GraphCodeBERT and UniXcoder models trained only on soft negatives)

import os
import json
import pandas as pd

df = pd.DataFrame()
df["rules"] = list(range(1,18+1))
col_name = {
    "codebert": "CodeBERT",
    "unixcoder": "UniXcoder",
    "graphcodebert": "GraphCodeBERT",
}
for model_type, col_name in col_name.items():
    path = f"data/{model_type}_rule_difficulties.json"
    scores = json.load(open(path))
    rule_diffs = []
    for i in range(1,18+1):
        rule = f"rule{i}"
        rule_diffs.append(scores[rule])
    df[col_name] = rule_diffs
save_path = "case_study/rule_difficulties.xlsx"
df.to_excel(save_path)