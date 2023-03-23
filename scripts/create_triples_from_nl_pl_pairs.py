#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import random
from typing import *
from tqdm import tqdm
from datautils.utils import * 
from collections import defaultdict

if __name__ == "__main__":
    nl_pl_pairs_path = sys.argv[1]
    k = int(sys.argv[2])
    stem, ext = os.path.splitext(nl_pl_pairs_path)
    dump_path = stem+f"_triples_{k}"+ext
    assert not os.path.exists(dump_path)
    # print(dump_path)
    intent_to_rec = defaultdict(lambda:[])
    nl_pl_pairs = json.load(open(nl_pl_pairs_path))
    
    for rec in nl_pl_pairs:
        nl = rec["intent"]
        intent_to_rec[nl].append(rec)
    intent_synsets = list(intent_to_rec.values())
    triples = []
    
    for i, rec_list in tqdm(enumerate(intent_synsets), total=len(intent_synsets)):
        for rec in rec_list:
            pool = flatten_list(intent_synsets[:i] + intent_synsets[i+1:])
            negs = [d["snippet"] for d in random.sample(pool, k=k)]
            pos = rec["snippet"]
            nl = rec["intent"]
            for neg in negs: triples.append((nl, pos, neg))
        
    print(f"created {len(triples)} for {len(nl_pl_pairs)} NL-PL pairs")
    with open(dump_path, "w") as f:
        json.dump(triples, f, indent=4)