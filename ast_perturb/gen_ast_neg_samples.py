#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Atharva Naik
import os
import json
import random
import argparse
import numpy as np
from typing import *
from tqdm import tqdm
from ast_perturb.ast_perturb2 import PerturbAst
# set logging level of transformers.
# transformers.logging.set_verbosity_error()
# seed
random.seed(0)
np.random.seed(0)
# torch.manual_seed(0)
# dataset options.
DATASETS = ["CoNaLa", "PyDocs"]
DATASETS_TRAIN_MAP = {
    "CoNaLa": "data/conala-mined.jsonl",
    "PyDocs": "external_knowledge/PyDocs_nl_pl_pairs.json",
}
# get arguments
def get_args():
    parser = argparse.ArgumentParser("script to train (using triplet margin loss), evaluate and predict with the CodeBERT in Late Fusion configuration for Neural Code Search.")
    parser.add_argument("-d", "--dataset", type=str, default="CoNaLa", 
                        help=f"dataset to work with from: {DATASETS}")
    parser.add_argument("-ns", "--num_splits", type=int, default=4)
    parser.add_argument("-si", "--split_index", type=int, required=True)
    # parser.add_argument("-topk", "--topk", default=10, type=int,
    #                     help="no. of similar intents to be paired with each intent")
    # parser.add_argument("-bs", "--batch_size", type=int, default=64, help="batch size")
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

def get_snippets(path) -> List[str]:
    snippets = set()
    # iterate over NL-PL pairs.
    if path.endswith(".json"):
        data: List[dict] = json.load(open(path))
    elif path.endswith(".jsonl"):
        data: List[dict] = read_jsonl(path)
    # populate intents
    for rec in data:
        snippets.add(rec["snippet"])
        
    return sorted(list(snippets))

def round_list(l: list, k: int=3) -> list:
    if isinstance(l, (float, int)):
        return round(l, k)
    for i in range(len(l)):
        l[i] = round_list(l[i], k=k)
    
    return l

# main function
if __name__ == "__main__":
    args = get_args()
    assert args.dataset in DATASETS
    # load all the NL-PL data.
    path = DATASETS_TRAIN_MAP[args.dataset]
    snippets = get_snippets(path)
    # create AST perturber to generate negative samples.
    perturber = PerturbAst()
    perturber.init()
    # create a map of snippet to AST perturbed negative samples per snippet.
    snippet_ast_neg_map: Dict[str, List[str]] = {}
    # index the split.
    split_size = len(snippets) // args.num_splits
    print(f"num_splits: {args.num_splits}")
    print(f"split_index: {args.split_index}")
    print(f"snippets = snippets[{args.split_index*split_size} : {(args.split_index+1)*split_size}]")
    snippets = snippets[args.split_index*split_size : (args.split_index+1)*split_size]
    # generate perturbed AST samples for code snippets.
    pbar = tqdm(snippets, disable=not(args.use_tqdm))
    i = 0
    tot = 0
    for code in pbar:
        i += 1
        # candidates = perturber.generate(code)
        try: candidates = perturber.generate(code)
        except SyntaxError as e:
            print(e, code); candidates = []
        tot += len(candidates)
        pbar.set_description(f"avg AST neg samples: {(tot/i):.3f}")
        snippet_ast_neg_map[code] = candidates
    # statistics of the AST perturbation procedure.
    avg_neg_samples = np.mean([len(cands) for cands in snippet_ast_neg_map.values()])
    print(f"code snippets: {len(snippets)}")
    print(f"code snippets: {avg_neg_samples}")
    # serialize and save the snippet AST negative samples map.
    path = f"{args.dataset}_AST_neg_samples_{args.num_splits}_{args.split_index+1}.json"
    with open(path, "w") as f:
        json.dump(snippet_ast_neg_map, f)
# python ast_perturb/gen_ast_neg_sampes.py -d CoNaLa -ns 4 -si 0 -tqdm