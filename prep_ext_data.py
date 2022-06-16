#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pathlib
from typing import *
from tqdm import tqdm
import os, sys, random, json
from datautils.utils import * 
    
# read  all the jsonl data in the external_knowledge folder.
data = []
posts = {}
for path in os.listdir("external_knowledge"):
    if not path.endswith(".jsonl"): continue
    print(f"\x1b[33m{path}\x1b[0m")
    path = os.path.join("external_knowledge", path)
    data += read_jsonl(path)
print(len(data))
for rec in data:
    k = rec["intent"]
    try: posts[k].append(rec)
    except KeyError: posts[k] = [rec]
print(len(posts))
print(list(posts.keys())[0])
posts = shuffle_dict(posts)
print("#"*os.get_terminal_size().columns)
print(len(index_dict(posts, 0)))
# create query_and_candidates and candidate snippets
test_posts = index_dict(posts, 0, 365)
train_val_posts = index_dict(posts, 365)

cands = {"snippets": []}
query_map = {}
cand_to_query = {}
for rec_list in test_posts.values():
    for rec in rec_list:
        cand_to_query[rec["snippet"]] = (rec["intent"], rec["snippet"]) 
        
for i, cand in enumerate(cand_to_query):
    q, c = cand_to_query[cand]
    try: query_map[q].append(i)
    except KeyError: query_map[q] = [i]
    cands["snippets"].append(c)

queries = []
for q, doc_list in query_map.items():
    queries.append({'query': q, 'docs': doc_list})
print(queries[1])
print(cands["snippets"][1])
# write the query and candidate files.
q_path = os.path.join("external_knowledge", "queries.json")
c_path = os.path.join("external_knowledge", "candidates.json")
with open(q_path, "w") as f:
    json.dump(queries, f)
with open(c_path, "w") as f:
    json.dump(cands, f)
print(len(queries), len(cands["snippets"]))

def main(posts: List[List[dict]], mode: str="default"):
    # print(posts[list(posts.keys())[0]])
    singleton_samples = 0
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
    print(f"found {singleton_samples} singleton samples (posts with only 1 answer)")  
    
    return triples

# main function.
if __name__ == "__main__":
    # mode = "rel_thresh_intra_categ_neg" # "intra_categ_neg"
    os.makedirs("triples", exist_ok=True)
    try: mode = sys.argv[1]
    except IndexError: mode = "default"
    # "rel_thresh_intra_categ_neg" # "default" # "rel_thresh"
    TYPE = "external_knowledge"
    if mode == "default": 
        triples_path: str = f"triples/triples_{TYPE}.json"
    else: 
        triples_path = os.path.join(
            "triples", f"triples_{TYPE}_{mode}.json"
        )
    triples = main(list(train_val_posts.values()), mode=mode)
    val_ratio: int=0.2
    val_size = int(len(triples)*val_ratio)
    
    stem, ext = os.path.splitext(triples_path)
    random.shuffle(triples)
    
    train_path = stem + "_train" + ext
    val_path = stem + "_val" + ext 
    print(train_path)
    print(val_path)
    
    print(f"len(train_data) = {len(triples[val_size:])}")
    print(f"len(val_data) = {len(triples[:val_size])}")  
    train_data = triples[val_size:]
    val_data = triples[:val_size]
    
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=4)
    with open(val_path, "w") as f:
        json.dump(val_data, f, indent=4)