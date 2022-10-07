#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from typing import *
from torch.utils.data import Dataset
from models.CodeBERT import CodeBERTripletNet
from models.UniXcoder import UniXcoderTripletNet
from models.GraphCodeBERT import GraphCodeBERTripletNet
from torchmetrics.functional import pairwise_cosine_similarity

# parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp", required=True, 
                    help="experiment/model to be loaded")
parser.add_argument("-p", "--path", type=str, 
                    default="data/analogy_test.json", 
                    help="path to the analogy dataset")
parser.add_argument("-bs", "--batch_size", default=48, 
                    help="batch size used while encoding code")
parser.add_argument("-m", "--model_type", type=str, 
                    required=True, help="the type of the model")
parser.add_argument("-id", "--device_id", type=str, 
                    default="cuda:0", help="GPU device ID to be used")
args = parser.parse_args()
# get path to tokenizer.
def get_tok_path(model_name: str) -> str:
    assert model_name in ["codebert", "graphcodebert", "unixcoder"]
    if model_name == "codebert":
        tok_path = os.path.expanduser("~/codebert-base-tok")
        if not os.path.exists(tok_path):
            tok_path = "microsoft/codebert-base"
    elif model_name == "graphcodebert":
        tok_path = os.path.expanduser("~/graphcodebert-base-tok")
        if not os.path.exists(tok_path):
            tok_path = "microsoft/grapcodebert-base"
    elif model_name == "unixcoder":
        tok_path = os.path.expanduser("~/unixcoder-base-tok")
        if not os.path.exists(tok_path):
            tok_path = "microsoft/unixcoder-base"
            
    return tok_path
# create model object and load checkpoint.
tok_path = get_tok_path(args.model_type)
if args.model_type == "unixcoder":
    model = UniXcoderTripletNet()
elif args.model_type == "codebert":
    model = CodeBERTripletNet(tok_path=tok_path)
elif args.model_type == "graphcodebert":
    model = GraphCodeBERTripletNet(tok_path=tok_path)
# load state dict into the model.
model.load_state_dict(torch.load(os.path.join(
    args.exp, "model.pt"), 
    map_location="cpu"
))
model.to(args.device_id)
analogy_data = json.load(open(args.path))
a = [i["a"] for i in analogy_data]
b = [i["b"] for i in analogy_data]
c = [i["c"] for i in analogy_data]
d = [i["d"] for i in analogy_data]

a = model.encode_emb(a, mode="code", batch_size=args.batch_size, 
                     use_tqdm=True, device=args.device_id)
b = model.encode_emb(b, mode="code", batch_size=args.batch_size, 
                     use_tqdm=True, device=args.device_id)
c = model.encode_emb(c, mode="code", batch_size=args.batch_size, 
                     use_tqdm=True, device=args.device_id)
d = model.encode_emb(d, mode="code", batch_size=args.batch_size, 
                     use_tqdm=True, device=args.device_id)
a = torch.stack(a)
b = torch.stack(b)
c = torch.stack(c)
d = torch.stack(d)
# dists = torch.cdist(c+b-a, d, p=2)
dists = -pairwise_cosine_similarity(c+b-a, d)
doc_ranks = dists.argsort(axis=1)
# recall@5 for analogy test.
# correct documents are range(len(d))
def compute_analogy_scores(doc_ranks, k=5, num_rules=9): # basically recall@5
    tot_score = 0
    rule_scores = [0 for i in range(num_rules)]
    rule_group_size = len(doc_ranks) // num_rules 
    for i, rank_list in enumerate(doc_ranks.tolist()):
        if i in rank_list[:5]:
            tot_score += 1
            rule_scores[i//200] += 1
    for j in range(len(rule_scores)):
        rule_scores[j] /= rule_group_size
        
    return tot_score/len(doc_ranks), rule_scores
# print(doc_ranks.shape)
analogy_score, rule_scores = compute_analogy_scores(doc_ranks, 5)
print(f"analogy score: {100*analogy_score:.3f}")
for i in range(9):
    print(f"rule{i+1} score: {100*rule_scores[i]:.3f}")