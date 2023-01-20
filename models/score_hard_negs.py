#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Atharva Naik (18CS10067)
import os
import json
import torch
from typing import *
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from models.CodeBERT import CodeBERTripletNet
from models.UniXcoder import UniXcoderTripletNet
from models.GraphCodeBERT import GraphCodeBERTripletNet

def load_model(model_type: str, tok_path: str, ckpt_path: str, device_id: str):
    """given model type, tokenizer path and checkpoint path, return 
    the triplet net instantiation with the correctly loaded checkpoint"""
    if model_type == "codebert":
        triplet_net = CodeBERTripletNet(tok_path=tok_path)
    elif model_type == "unixcoder":
        triplet_net = UniXcoderTripletNet(tok_path=tok_path)
    elif model_type == "graphcodebert":
        triplet_net = GraphCodeBERTripletNet(tok_path=tok_path)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    triplet_net.load_state_dict(state_dict)
    triplet_net.to(device_id)
    
    return triplet_net

def load_data(neg_path: str, anchor_p_path: str) -> Tuple[Dict[int, str], Dict[int, str], 
                                                          Dict[int, str], Tuple[int, int, int, str]]:
    i_n, i_p, i_a = 0, 0, 0
    anchor_texts = {}
    pos_codes, neg_codes = {}, {}
    nl_and_code = json.load(open(anchor_p_path))
    code_and_perturbations = json.load(open(neg_path))
    triplet_and_rule = []
    for _, rec in tqdm(enumerate(nl_and_code), 
                       total=len(nl_and_code)):
        nl = rec["intent"]
        pos = rec["snippet"]
        neg_and_rules = code_and_perturbations[pos]
        if len(neg_and_rules) == 0: continue
        elif len(neg_and_rules) == 1: continue # formatting mistake that we can ignore for now
        assert len(neg_and_rules[0]) == 2, "incorrectly formatted file"
        if pos not in pos_codes:
            pos_codes[pos] = i_p
            i_p += 1
        pi = pos_codes[pos]
        if nl not in anchor_texts:
            anchor_texts[nl] = i_a
            i_a += 1
        ai = anchor_texts[nl]
        for neg, rule in code_and_perturbations[pos]:
            if neg not in neg_codes:
                neg_codes[neg] = i_n
                i_n += 1
            ni = neg_codes[neg]
            triplet_and_rule.append((
                ai, pi, ni, rule,
            ))
    pos_codes = {v: k for k,v in pos_codes.items()}
    neg_codes = {v: k for k,v in neg_codes.items()}
    anchor_texts = {v: k for k,v in anchor_texts.items()}
    
    return anchor_texts, pos_codes, neg_codes, triplet_and_rule

model_ind = 2 # 0,1,2
anchor_p_path = "data/conala-mined-100k.json"
model_type = ["graphcodebert", "unixcoder", "codebert"][model_ind]
ckpt_path = ["experiments/GraphCodeBERT/model.pt", "experiments/UniXcoder/model.pt", "experiments/CodeBERT/model.pt"][model_ind]
tok_path = ["~/graphcodebert-base-tok", "~/unixcoder-base-tok", "~/codebert-base-tok"][model_ind]
device_id = "cuda:0"
neg_path = "CoNaLa_AST_neg_samples.json"
tok_path = os.path.expanduser(tok_path)
triplet_net = load_model(model_type, tok_path, 
                         ckpt_path, device_id)
a, p, n, T = load_data(neg_path, anchor_p_path)
intents = list(a.values())#[:500]
pos_codes = list(p.values())#[:500]
neg_codes = list(n.values())#[:500]
# print(len(T), T[0])
intents = triplet_net.encode_emb(intents, device_id=device_id,
                                 batch_size=64, use_tqdm=True,
                                 mode="text")
pos_codes = triplet_net.encode_emb(pos_codes, device_id=device_id,
                                   batch_size=64, use_tqdm=True,
                                   mode="code")
neg_codes = triplet_net.encode_emb(neg_codes, device_id=device_id,
                                   batch_size=64, use_tqdm=True,
                                   mode="code")
A = defaultdict(lambda:[])
P = defaultdict(lambda:[])
N = defaultdict(lambda:[])
print(a[T[0][0]], p[T[0][1]], n[T[0][2]], T[0][3])
# intents = torch.stack(intents)
# pos_codes = torch.stack(pos_codes)
# neg_codes = torch.stack(neg_codes)
pdist = nn.PairwiseDistance()
for ai, pi, ni, rule in tqdm(T):#[:10]):
    A[rule].append(intents[ai])
    P[rule].append(pos_codes[pi])
    N[rule].append(neg_codes[ni])
rulewise = {}
for rule in A:
    Ar = torch.stack(A[rule])
    Pr = torch.stack(P[rule])
    Nr = torch.stack(N[rule])
    d_apr = pdist(Ar, Pr)
    d_anr = pdist(Ar, Nr)
    tot = len(d_apr)
    matches = (d_apr < d_anr).sum().item()
    rulewise[rule] = (tot, matches, matches/tot)
for rule in rulewise:
    tot, matches, score = rulewise[rule]
    print(f"\x1b[32;1m{rule}:\x1b[0m {100*score:.3f} ({matches}/{tot})")