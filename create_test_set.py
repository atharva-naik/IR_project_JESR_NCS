#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# create test set for retrieval settings.
import json
# data = json.load(open("data/conala-test.json"))
data = json.load(open("data/codesearchnet-test.json"))

posts = {}
queries = {}
candidates = {}

query_id = 0
candi_id = 0

for rec in data:
    intent = rec["intent"]
    snippet = rec["snippet"]
    try: code_annot = rec['rewritten_intent']
    except KeyError: code_annot = None
    try: posts[intent].append((snippet, code_annot))
    except KeyError: posts[intent] = [(snippet, code_annot)]

for i, rec in enumerate(data):
    intent = rec["intent"]
    snippet = rec["snippet"]
    try: code_annot = rec['rewritten_intent']
    except KeyError: code_annot = None
    try:
        queries[intent]+1
    except KeyError:
        queries[intent] = query_id
        query_id += 1
    try:
        candidates[snippet][0]+1
    except KeyError:
        candidates[snippet] = (candi_id, code_annot)
        candi_id += 1
        
query_map = []
for query in queries:
    snippets = [i[0] for i in posts[query]]
    snippet_ids = [candidates[snippet][0] for snippet in snippets]
    query_map.append({"query": query, "docs": snippet_ids})
# print(candidates)
candi_records = {"snippets": [], "annotations": []}
candi_records["snippets"] = [code_snippet for code_snippet in candidates.keys()]
null_ctr = 0
for _, code_annotation in candidates.values():
    if code_annotation is None: 
        candi_records["annotations"].append("")
        null_ctr += 1
    else: candi_records["annotations"].append(code_annotation)
print(null_ctr)
del candi_records["annotations"]
with open("data/candidates_codesearchnet.json", "w") as f:
    json.dump(
        candi_records,        
        f, indent=4
    )
with open("data/queries_codesearchnet.json", "w") as f:
    json.dump(
        query_map, 
        f, indent=4
    )