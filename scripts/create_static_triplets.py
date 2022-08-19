#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import pprint
import random
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Script to create a static triples out of xNL-PL pairs.')
parser.add_argument('-p', '--path', type=str, required=True,
                    help="path to the JSON/JSONL file with NL-PL pairs")
parser.add_argument('-s', '--seed', type=int, default=42,
                    help="seed value for random split")
parser.add_argument('-r', '--neg_sample_ratio', type=int, default=10, 
                    help="number of negative samples per NL-PL pair.")
args = parser.parse_args()

random.seed(args.seed)
data = json.load(open(args.path))
all_PLS = [rec["snippet"] for rec in data]
triples = []
for i, rec in tqdm(enumerate(data), total=len(data)):
    NL = rec["intent"]
    PL = rec["snippet"]
    cands = all_PLS[:i]+all_PLS[i+1:]
    # print(len(cands))
    other_PLs = random.sample(cands, k=args.neg_sample_ratio)
    for PL_neg in other_PLs:
        triples.append((NL, PL, PL_neg))
fname_wout_ext, ext = os.path.splitext(args.path)
save_path = fname_wout_ext + "_triplets" + ext
print(len(triples))
with open(save_path, "w") as f:
    json.dump(triples, f, indent=4)