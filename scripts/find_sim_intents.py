#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Atharva Naik
import os
import json
import torch
import random
import argparse
import numpy as np
import transformers
from typing import *
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
# set logging level of transformers.
transformers.logging.set_verbosity_error()
# seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# dataset options.
DATASETS = ["CoNaLa", "PyDocs"]#, "CodeSearchNet"]
DATASETS_TRAIN_MAP = {
    "CoNaLa": "data/conala-mined-100k_train.json",
    "PyDocs": "external_knowledge/PyDocs_nl_pl_pairs_train.json",
    # "CodeSearchNet": "data/codesearchnet.json",
}
# get arguments
def get_args():
    parser = argparse.ArgumentParser("script to train (using triplet margin loss), evaluate and predict with the CodeBERT in Late Fusion configuration for Neural Code Search.")
    parser.add_argument("-td", "--target_dataset", type=str, default="CoNaLa", 
                        help=f"dataset to work with from: {DATASETS}")
    parser.add_argument("-d", "--device", type=str, default="cpu", 
                        help="device string (GPU) for doing training/testing")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-topk", "--topk", default=10, type=int,
                        help="no. of similar intents to be paired with each intent")
    parser.add_argument("-tqdm", "--use_tqdm", action="store_true",
                        help="show tqdm progress bar during inference")

    return parser.parse_args()

def read_jsonl(path: str) -> List[dict]:
    """read data into list of dicts from a .jsonl file"""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            data.append(json.loads(line))
            
    return data

def get_intent_map(path) -> Dict[str, int]:
    intents: Dict[str, int] = {}
    # iterate over NL-PL pairs.
    if path.endswith(".json"):
        data: List[dict] = json.load(open(path))
    elif path.endswith(".jsonl"):
        data: List[dict] = read_jsonl(path)
    # populate intents
    for rec in data:
        intent = rec["intent"] 
        if intent not in intents:
            intents[intent] = len(intents)
        
    return intents

def round_list(l: list, k: int=3) -> list:
    if isinstance(l, (float, int)):
        return round(l, k)
    for i in range(len(l)):
        l[i] = round_list(l[i], k=k)
    
    return l

# main function
if __name__ == "__main__":
    args = get_args()
    assert args.target_dataset in DATASETS
    # load all the NL-PL data.
    path = DATASETS_TRAIN_MAP[args.target_dataset]
    intents = get_intent_map(path)
    # create a map of similar intents based 
    sim_intents_map: Dict[int, List[Tuple[int, float]]] = {}
    sbert = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
    # move to correct device.
    sbert.to(args.device)
    emb_matrix = torch.as_tensor(sbert.encode(
        list(intents.keys()), device=args.device, 
        show_progress_bar=args.use_tqdm, 
        batch_size=args.batch_size,
    ))
    J = torch.ones(len(intents), len(intents)) - torch.eye(len(intents))
    J = J.to(emb_matrix.device)
    scores = J*util.dot_score(emb_matrix, emb_matrix)
    sim_scores = round_list(torch.topk(scores, k=args.topk, axis=1).values.tolist())
    sim_ranks = round_list(torch.topk(scores, k=args.topk, axis=1).indices.tolist())
    sim_intents: Dict[str, List[Tuple[str, float]]] = {}
    intent_list = list(intents.keys())
    for i in range(len(sim_scores)):
        sim_intents[intent_list[i]] = []
        for j in range(len(sim_scores[i])):
            sim_intents[intent_list[i]].append((
                intent_list[sim_ranks[i][j]],
                sim_scores[i][j],
            ))
    save_path = f"{args.target_dataset}_top{args.topk}_sim_intents.json"
    with open(save_path, "w") as f:
        json.dump(sim_intents, f)
# scripts/find_sim_intents.py -d "cuda:1" -bs 64 -tqdm -td CoNaLa