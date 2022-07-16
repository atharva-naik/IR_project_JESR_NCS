#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Atharva Naik
import os
import json
import time
import torch
import models
import random
import pathlib
import argparse
import numpy as np
import transformers
from typing import *
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from datautils import read_jsonl
from sklearn.metrics import ndcg_score as NDCG
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from models.metrics import TripletAccuracy, recall_at_k
from sklearn.metrics import label_ranking_average_precision_score as MRR
from models import test_ood_performance, get_tok_path, dynamic_negative_sampling

# set logging level of transformers.
transformers.logging.set_verbosity_error()
# seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# dataset options.
MODELS = ["codebert", "graphcodebert", "unixcoder"]
DATASETS = ["CoNaLa", "PyDocs", "CodeSearchNet"]
DATASETS_TRAIN_MAP = {
    "CoNaLa": "data/conala-mined.jsonl",
    "PyDocs": "external_knowledge/PyDocs_nl_pl_pairs.json",
    "CodeSearchNet": "data/codesearchnet.json",
}
# get arguments
def get_args():
    parser = argparse.ArgumentParser("script to train (using triplet margin loss), evaluate and predict with the CodeBERT in Late Fusion configuration for Neural Code Search.")
    parser.add_argument("-m", "--model", type=str, default="codebert")
    parser.add_argument("-td", "--target_dataset", type=str, default="CoNaLa", 
                        help=f"dataset to work with from: {DATASETS}")
    parser.add_argument("-d", "--device_id", type=str, default="cpu", 
                        help="device string (GPU) for doing training/testing")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("-ckpt", "--checkpoint_path", type=str, required=True)
    parser.add_argument("-tqdm", "--use_tqdm", action="store_true",
                        help="show tqdm progress bar during inference")

    return parser.parse_args()

def load_nl_pl_data(path) -> Tuple[List[str], List[str], List[Tuple[int, int]], Dict[int, int]]:
    intents = {}
    snippets = {}
    nl_pl_pairs = []
    snippet_to_intent_map = {} 
    # iterate over NL-PL pairs.
    if path.endswith(".json"):
        data: List[dict] = json.load(open(path))
    elif path.endswith(".jsonl"):
        data: List[dict] = read_jsonl(path)
    # populate intents, snippets and NL-PL pairs.
    for rec in data:
        intent = rec["intent"] 
        snippet = rec["snippet"]
        if intent not in intents:
            intents[intent] = len(intents)
        if snippet not in snippets:
            snippets[snippet] = len(snippets)
    for rec in data:
        intent = rec["intent"] 
        snippet = rec["snippet"]
        intent_id = intents[intent]
        snippet_id = snippets[snippet]
        snippet_to_intent_map[snippet_id] = intent_id 
        nl_pl_pairs.append((intent_id, snippet_id))
        
    return intents, snippets, nl_pl_pairs, snippet_to_intent_map

def buffer_intents_and_snippets(model, intents: Dict[str, int], 
                                snippets: Dict[str, int], args):
    intent_write_path = f"{args.model}_{args.target_dataset}_intents.pt"
    snippet_write_path = f"{args.model}_{args.target_dataset}_snippets.pt"
    embeds = torch.stack(triplet_net.encode_emb(
        list(intents.keys()), 
        mode="text", **vars(args)
    ))
    torch.save(embeds, intent_write_path)
    embeds = torch.stack(triplet_net.encode_emb(
        list(snippets.keys()), 
        mode="code", **vars(args)
    ))
    torch.save(embeds, snippet_write_path)
    
# main function
if __name__ == "__main__":
    args = get_args()
    assert args.model in MODELS
    assert args.target_dataset in DATASETS
    # load all the NL-PL data.
    path = DATASETS_TRAIN_MAP[args.target_dataset]
    intents, snippets, nl_pl_pairs, snippet_to_intent_map = load_nl_pl_data(path)
    # transformer to encode NL-PL pairs.
    tok_path = get_tok_path(args.model)
    if args.model == "codebert":
        from models.CodeBERT import CodeBERTripletNet
        triplet_net = CodeBERTripletNet(tok_path=tok_path, **vars(args))
    elif args.model == "graphcodebert":
        from models.GraphCodeBERT import GraphCodeBERTripletNet
        triplet_net = GraphCodeBERTripletNet(tok_path=tok_path, **vars(args))
    # load checkpoint path.
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    triplet_net.load_state_dict(state_dict)
    print(f"loaded state_dict from: {args.checkpoint_path}")
    # encode NL-PL pairs.
    buffer_intents_and_snippets(
        triplet_net, intents, 
        snippets, args,
    )