#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
from typing import *
# from tqdm import tqdm
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Script to create a static train-val split from NL-PL pairs.')
parser.add_argument('-p', '--path', type=str, required=True,
                    help="path to the JSON/JSONL file with NL-PL pairs")
parser.add_argument('-s', '--seed', type=int, default=42,
                    help="seed value for random split")
parser.add_argument('-r', '--split_ratio', type=float, default=0.2, 
                    help="fractional relative size of val set")
args = parser.parse_args()

def read_jsonl(path: str) -> List[dict]:
    """
    read data into list of dicts from a .jsonl file
    """
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            data.append(json.loads(line))
            
    return data

if args.path.endswith(".json"):
    nl_pl_pairs: List[dict] = json.load(open(args.path))
elif args.path.endswith(".jsonl"):
    nl_pl_pairs: List[dict] = read_jsonl(args.path)
print(len(nl_pl_pairs))
X_train, X_val, _, _ = train_test_split(
    nl_pl_pairs, range(len(nl_pl_pairs)), 
    random_state=args.seed, test_size=args.split_ratio, 
)
print(len(X_train), len(X_val))
print(X_train[0])
print(X_val[0])

fname_wout_ext, ext = os.path.splitext(args.path)
train_path = fname_wout_ext + "_train" + ext
val_path = fname_wout_ext + "_val" + ext
with open(train_path, "w") as f:
    json.dump(X_train, f)
with open(val_path, "w") as f:
    json.dump(X_val, f)