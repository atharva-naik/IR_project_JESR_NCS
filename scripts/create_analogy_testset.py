#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json
import random
import itertools
from typing import *
from tqdm import tqdm
from pprint import pprint

# the number of test cases per rule category in the analogy test.
try: N = sys.argv[1]
except IndexError: N = 200

path = "CoNaLa_AST_neg_samples_4_{}.json"
data: List[Dict[str, List[Tuple[str, str]]]] = [json.load(open(path.format(i+1))) for i in range(3)]
rule_wise_transforms_pairs: Dict[str, List[Tuple[str, str]]] = {f"rule{i+1}": [] for i in range(9)}
for i in range(len(data)):
    for orig, transforms in tqdm(data[i].items()):
        for rule_transform_pair in transforms:
            if isinstance(rule_transform_pair, list):
                transformed = rule_transform_pair[0]
                rule = rule_transform_pair[1]
                if len(transformed) > 30 and len(orig) > 30:
                    rule_wise_transforms_pairs[rule].append((
                        orig, transformed,
                    ))
                    
analogy_testset = []
for rule, transform_pairs in rule_wise_transforms_pairs.items():
    print(f"{rule}: ", len(transform_pairs))
    # N=200 analogies per rule category:
    # sample pairs of transform pairs for analogy test.
    random.seed(42)
    indices = random.sample(range(len(transform_pairs)), 100)
    ab_cd = itertools.combinations(indices, 2)
    random.seed(42)
    pbar = tqdm(random.sample(list(ab_cd), N))
    for i,j in pbar:
        a,b = transform_pairs[i]
        c,d = transform_pairs[j]
        record = {"a":a, "b":b, "c":c, "d":d}
        pbar.set_description(f"i:{i} j:{j}")
        analogy_testset.append(record)
with open("data/analogy_test.json", "w") as f:
    json.dump(analogy_testset, 
              f, indent=4)