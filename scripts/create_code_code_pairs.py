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

if __name__ == "__main__":
    train_data = json.load(open("./data/conala-mined-100k_train.json"))
    code_pairs = create_code_code_pairs(train_data)
    # code-code pairs.
    with open("./data/conala-mined-100k_train_ccp.json", "w") as f:
        json.dump(code_pairs, f, indent=4)