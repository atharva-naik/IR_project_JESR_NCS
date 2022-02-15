#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
from tqdm import tqdm
from typing import List
from datautils.utils import * 
# from datautils import TextCodeTriplets

random.seed(2022)
def main(data_path: str, triples_path: str):
    data: List[dict] = read_jsonl(data_path)
    posts: List[List[dict]] = list(get_posts(data).values())
    if os.path.exists(triples_path):
        triples = json.load(open(triples_path))
        return triples
    triples = create_triples(posts)
        # if i == 10: break # DEBUG.
    print(f"caching data at {triples_path}")
    with open(triples_path, "w") as f:
        json.dump(triples, f, indent=4)
    print(f"found {singleton_samples} singleton samples (posts with only 1 answer)")  
    
    return triples

    
if __name__ == "__main__":
    triples_path="triples.json"
    triples = main(data_path="data/conala-mined.jsonl", 
                   triples_path=triples_path)
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