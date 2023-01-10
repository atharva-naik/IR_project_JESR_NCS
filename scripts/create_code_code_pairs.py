#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Atharva Naik (18CS10067)

# create code-code pairs for the CodeRetriever objective
import json
import itertools
from typing import *
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

def create_code_synsets(data: List[dict]) -> Tuple[Dict[str, List[Tuple[str, int]]], Dict[str, str]]:
    code_synsets = defaultdict(lambda:[])
    code_to_intent = {} # key calculation map.
    for rec in data:
        code_synsets[rec['intent']].append((
            rec['snippet'], rec['prob']
        ))
        code_to_intent[rec['snippet']] = rec['intent']

    return code_synsets, code_to_intent

class CodeSynsets:
    def __init__(self, path: str):
        self.path = path
        with open(path) as f: data = json.load(f)
        self.code_synsets = data['code_synsets']
        self.code_to_intent = data['code_to_intent']
        
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