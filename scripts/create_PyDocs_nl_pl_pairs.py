#!/usr/bin/env python3
import os
import json
from typing import *
from tqdm import tqdm

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
data = {}
test_queries = [i['query'] for i in json.load(open(os.path.join("external_knowledge", "queries.json")))]
for k in [1, 5]:
    for t in ["intent", "snippet"]:
        path = os.path.join("external_knowledge", f"goldmine_{t}_count100k_topk{k}_temp2.jsonl")
        items = read_jsonl(path)
        for item in tqdm(items):
            intent = item["intent"]
            snippet = item["snippet"]
            if intent in test_queries: continue
            data[f"{intent}#-#{snippet}"] = {"intent": intent, "snippet": snippet}
print(f"dataset has {len(data)} unique NL-PL pairs that don't coincide with the test set.")
with open("external_knowledge/PyDocs_nl_pl_pairs.json", "w") as f:
    json.dump(list(data.values()), f)