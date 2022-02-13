#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import random
from typing import List, Dict


def read_jsonl(path: str) -> List[dict]:
    """
    read data into list of dicts from a .jsonl file
    """
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            data.append(json.loads(line))
            
    return data

def get_posts(data: List[dict]) -> Dict[str, dict]:
    posts = {} # group the posts by the intent (or post title.)
    # the value contains a list of dataset entries featuring relevant code snippets in the decreasing order of relevance.
    for rec in data:
        intent = rec["intent"]
        try: posts[intent].append(rec)
        except KeyError: posts[intent] = [rec]
            
    return posts

def get_list_complement(l: list, i: int):
    nl = []
    for j in range(len(l)):
        if i == j: continue
        nl.append(l[j])
        
    return nl

def sample_list(l: list, k: int=1):
    """get sampled list."""
    sampled_list = []
    for i in random.sample(range(len(l)), k=k):
        sampled_list.append(l[i])
    
    return sampled_list

def flatten_list(l: List[list]) -> list:
    flat_list = []
    for sub_list in l:
        flat_list += sub_list
        
    return flat_list
        