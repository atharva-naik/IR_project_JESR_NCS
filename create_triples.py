#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
from tqdm import tqdm
from typing import List
from datautils.utils import * 
# from datautils import TextCodeTriplets

random.seed(2022)
# valid modes: ['default', 'rel_thresh']
def main(data_path: str, triples_path: str, mode: str="default"):
    data: List[dict] = read_jsonl(data_path)
    posts: List[List[dict]] = list(get_posts(data).values())
    singleton_samples = 0
    if os.path.exists(triples_path):
        triples = json.load(open(triples_path))
        return triples
    if mode == "default":
        triples = create_triples(posts, neg_to_pos_ratio=3)
    elif mode == "rel_thresh":
        triples = create_relevant_triples(
            posts, neg_to_pos_ratio=3,
            pos_rel_rank_thresh=0.25
        )
    elif mode == "intra_categ_neg":
        triples = create_triples_intra_categ_neg(
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
    print(f"caching data at {triples_path}")
    with open(triples_path, "w") as f:
        json.dump(triples, f, indent=4)
    print(f"found {singleton_samples} singleton samples (posts with only 1 answer)")  
    
    return triples

    
if __name__ == "__main__":
    mode = "rel_thresh_intra_categ_neg" # "intra_categ_neg"
    # "rel_thresh_intra_categ_neg" # "default" # "rel_thresh"
    if mode == "default": triples_path: str = "triples.json"
    else: triples_path = f"triples_{mode}.json"
    triples = main(data_path="data/conala-mined.jsonl", 
                   triples_path=triples_path, mode=mode)
    val_ratio: int=0.2
    val_size = int(len(triples)*val_ratio)
    stem, ext = os.path.splitext(triples_path)
    random.shuffle(triples)
    train_path = stem + "_train" + ext
    val_path = stem + "_val" + ext
    
    train_data = triples[val_size:]
    val_data = triples[:val_size]
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=4)
    with open(val_path, "w") as f:
        json.dump(val_data, f, indent=4)