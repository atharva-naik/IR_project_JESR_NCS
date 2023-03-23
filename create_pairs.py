#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import random
from tqdm import tqdm
from typing import List
from datautils.utils import * 

random.seed(2022)
# valid modes: ['default', 'rel_thresh']
def main(data_path: str, pairs_path: str):
    data: List[dict] = read_jsonl(data_path)
    posts: List[List[dict]] = list(get_posts(data).values())
    if os.path.exists(pairs_path):
        pairs = json.load(open(pairs_path))
        return pairs
    pairs = create_pairs(posts, neg_to_pos_ratio=1)
    print(f"caching data at {pairs_path}")
    with open(pairs_path, "w") as f:
        json.dump(pairs, f, indent=4)
    # print(f"found {singleton_samples} singleton samples (posts with only 1 answer)")  
    return pairs

    
if __name__ == "__main__":
    os.makedirs("triples", exist_ok=True)
    pairs_path = "triples/nl_code_pairs.json"
    pairs = main(data_path="data/conala-mined.jsonl", 
                 pairs_path=pairs_path)
    print(f"dataset has {len(pairs)} NL-PL pairs")
    
    val_ratio: int=0.2
    val_size = int(len(pairs)*val_ratio)
    stem, ext = os.path.splitext(pairs_path)
    random.shuffle(pairs)
    
    train_path = stem + "_train" + ext
    val_path = stem + "_val" + ext
    train_data = pairs[val_size:]
    val_data = pairs[:val_size]
    
    print(train_path)
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=4)
    # with open(val_path, "w") as f:
    #     json.dump(val_data, f, indent=4)