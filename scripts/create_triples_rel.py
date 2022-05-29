#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import random
from tqdm import tqdm
from typing import List
from datautils.utils import * 
# from datautils import TextCodeTriplets

random.seed(2022)
os.makedirs("triples", exist_ok=True)
# triples, train and val paths.
triples_path = os.path.join("triples", "triples_w_rel.json")
train_path = os.path.join("triples", "triples_w_rel_train.json")
val_path = os.path.join("triples", "triples_w_rel_val.json")
# don't overwrite files.
if os.path.exists(triples_path) and os.path.exists(train_path) and os.path.exists(val_path):
    exit(f"""all of the files: [{triples_path}, {train_path}, {val_path}] 
         exist already, so they won't be overwritten!""")

data = read_jsonl("data/conala-mined.jsonl")
triples: List[dict] = []
posts: List[List[dict]] = list(get_posts(data).values())
singleton_samples: int = 0
id = 0

for i, associated_snippets in tqdm(enumerate(posts), 
                                   desc="creating triplets", 
                                   total=len(posts)):
    # soft/easier to learn negatives.
    soft_negatives = flatten_list(
        get_list_complement(posts, i)
    )
    a = associated_snippets[0]["intent"]
    for item in associated_snippets:
        p = item["snippet"]
        r_ap = item["prob"]
        # sample 3 soft negatives per +ve sample.
        for item_ in sample_list(soft_negatives, k=3):
            n = item_["snippet"]
            r_an = -1 # to denote that this is value is invalid and needs to be calculated.
            triples.append({
                            "id": id, "p": p, 
                            "n": n, "a": a, 
                            "r_ap": r_ap, 
                            "r_an": r_an,
                           })
            id += 1            
        # sample hard negatives having relevance diff of 0.3
        for item_ in associated_snippets:
            if not(item_["prob"] < (r_ap - 0.3)):
                continue
            n = item_["snippet"]
            r_an = item_["prob"]
            triples.append({
                            "id": id, "p": p, 
                            "n": n, "a": a, 
                            "r_ap": r_ap, 
                            "r_an": r_an,
                           })
            id += 1
    # if i == 1000: break
# create train & validation datasets.
val_ratio: int=0.2
val_size = int(len(triples)*val_ratio)
stem, ext = os.path.splitext(triples_path)
# # randomly shuffle triples.
# random.shuffle(triples)
train_data = triples[val_size:]
val_data = triples[:val_size]
# write data.
with open(train_path, "w") as f:
    json.dump(train_data, f, indent=4)
with open(val_path, "w") as f:
    json.dump(val_data, f, indent=4)