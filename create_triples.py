#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import random
import pathlib
import argparse
from typing import *
from tqdm import tqdm
from datautils.utils import * 
# from datautils import TextCodeTriplets
random.seed(2022)
def filt_topk_by_rel(data: List[dict], k: int=10**5):
    """filter top k examples by relevance scores."""
    return sorted(data, reverse=True, key=lambda x: x["prob"])[:k]

def get_args():
    MODES = ["default", "fixed", "rel_thresh", "rel_thresh_fixed", 
             "intra_categ_neg", "intra_categ_neg_fixed", 
             "rel_thresh_intra_categ_neg"]
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="data/conala-mined.jsonl", 
                        help="path to the dataset. Defaults to: data/conala-mined.jsonl")
    parser.add_argument("-m", "--mode", type=str, default="default", 
                        help="triplet generation mode out of: {}".format(MODES))
    parser.add_argument("-vr", "--val_ratio", type=float, default=0.2, 
                        help="size of validation set relative to train")
    parser.add_argument("-fr", "--filt_rel", action="store_true",
                        help="filter top-k by relevance")
    parser.add_argument("-k", "--k", type=int, default=10**5,
                        help="k for top-k filtering by relevance")
    args = parser.parse_args()

    return args
    
# valid modes: ['default', 'rel_thresh']
def main(data_path: str, triples_path: str, **args):
    _, ext = os.path.splitext(data_path)
    k = args.get("k", 10**5)
    filt_rel = args.get("filt_rel", False)
    if ext == ".jsonl":
        data: List[dict] = read_jsonl(data_path)
    else: data: List[dict] = json.load(open(data_path))
    if filt_rel:
        data: List[dict] = filt_topk_by_rel(data, k=k)
    posts: List[List[dict]] = list(get_posts(data).values())
    singleton_samples = 0
    if os.path.exists(triples_path):
        triples = json.load(open(triples_path))
        return triples
    if mode == "default":
        triples = create_triples(posts, neg_to_pos_ratio=3)
    elif mode == "fixed":
        triples = create_triples_fixed(posts, neg_to_pos_ratio=3)
    elif mode == "rel_thresh":
        triples = create_relevant_triples(
            posts, neg_to_pos_ratio=3,
            pos_rel_rank_thresh=0.25
        )
    elif mode == "rel_thresh_fixed":
        triples = create_relevant_triples_fixed(
            posts, neg_to_pos_ratio=3,
            pos_rel_rank_thresh=0.25
        )
    elif mode == "intra_categ_neg":
        triples = create_triples_intra_categ_neg(
            posts, neg_to_pos_ratio=3,
            intra_categ_thresh=0.3,
        )
    elif mode == "intra_categ_neg_fixed":
        triples = create_triples_intra_categ_neg_fixed(
            posts, neg_to_pos_ratio=3,
            intra_categ_thresh=0.3,
        )
    elif mode == "rel_thresh_intra_categ_neg":
        triples = create_relevant_triples_intra_categ_neg(
            posts, neg_to_pos_ratio=3,
            pos_rel_rank_thresh=0.25,
            intra_categ_thresh=0.2,
        )
        # if i == 10: break # DEBUG.
    print(f"found {singleton_samples} singleton samples (posts with only 1 answer)")  
    # print(f"caching data at {triples_path}")
    # with open(triples_path, "w") as f:
    #     json.dump(triples, f, indent=4)
    return triples

# main function.
if __name__ == "__main__":
    # mode = "rel_thresh_intra_categ_neg" # "intra_categ_neg"
    os.makedirs("triples", exist_ok=True)
    args = get_args()
    mode = args.mode # triple generation mode (algo).
    data_path = args.path # source path for triple gen.
    if not os.path.exists(data_path): 
        raise FileNotFoundError(data_path)
    TYPE = pathlib.Path(data_path).stem
    if mode == "default": 
        triples_path: str = f"triples_{TYPE}.json"
    else: 
        triples_path: str = f"triples_{TYPE}_{mode}.json"
    triples_path = os.path.join("triples", triples_path)
    stem, ext = os.path.splitext(triples_path)
    val_path = stem+"_val"+ext
    train_path = stem+"_train"+ext
    triples = main(data_path=data_path, 
                   triples_path=triples_path, 
                   **vars(args))
    val_size=int(len(triples)*args.val_ratio)
    with open(train_path, "w") as f:
        json.dump(triples[val_size:], f, indent=4)
    with open(val_path, "w") as f:
        json.dump(triples[:val_size], f, indent=4)