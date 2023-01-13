#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Atharva Naik (18CS10067)

# create code-code pairs for the CodeRetriever objective
import json
import itertools
import numpy as np
from typing import *
from fuzzywuzzy import fuzz
from collections import defaultdict

def create_code_code_pairs(data: List[dict]) -> List[Tuple[str, str]]:
    intent_to_codes = defaultdict(lambda:[])
    for rec in data:
        intent_to_codes[rec["intent"]].append(rec['snippet'])
    code_pairs = []
    for intent, codes in intent_to_codes.items():
        for code_pair in itertools.combinations(codes, r=2):
            code_pairs.append(code_pair)

    return code_pairs

def create_code_synsets(data: List[dict]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, str]]:
    code_synsets = defaultdict(lambda:{})
    code_to_intent = {} # key calculation map.
    for rec in data:
        code_synsets[rec['intent']][rec['snippet']] = rec['prob']
        code_to_intent[rec['snippet']] = rec['intent']

    return code_synsets, code_to_intent

class CodeSynsets:
    def __init__(self, path: str):
        self.path = path
        with open(path) as f: data = json.load(f)
        self.code_synsets = data['code_synsets']
        self.code_to_intent = data['code_to_intent'] 
        
    def pick_lex(self, code: str):
        res = []
        syns = self[code]
        q_rel = syns[code]
        q = " ".join(code.replace("_", " ").split())
        for cand in syns:
            d = " ".join(cand.replace("_", " ").split())
            if d == q: continue
            lex_sim = (fuzz.token_sort_ratio(q, d))/100
            res.append((cand, lex_sim))
        if len(res) == 0: return code
        return res[np.argmax([i for _,i in res])][0]       
        
    def pick_rel_lex(self, code: str):
        res = []
        syns = self[code]
        q_rel = syns[code]
        q = " ".join(code.replace("_", " ").split())
        for cand, rel in syns.items():
            if rel <= q_rel: continue # if d == q: continue
            d = " ".join(cand.replace("_", " ").split())
            lex_sim = (fuzz.token_sort_ratio(q, d))/100
            # res.append((d, rel-lex_sim))
            res.append((d, rel/lex_sim))
        if len(res) == 0: return code
        return res[np.argmax([i for _,i in res])][0]   
        
    def pick_rel(self, code: str):
        syns = self[code]
        score = syns[code]
        for code, rel in syns.items():
            if rel > score: return code
        return code
        
    def __getitem__(self, code: str):
        intent = self.code_to_intent[code]
        return self.code_synsets[intent]
    
# main method.
if __name__ == "__main__":
    train_data = json.load(open("./data/conala-mined-100k_train.json"))
    # code_pairs = create_code_code_pairs(train_data)
    code_synsets, code_to_intent = create_code_synsets(train_data)
    # code-code pairs.
    with open("./data/conala-mined-100k_train_csyn.json", "w") as f:
        json.dump({
            "code_synsets": code_synsets,
            "code_to_intent": code_to_intent,
        }, f, indent=4)