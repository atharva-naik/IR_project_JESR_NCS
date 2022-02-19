#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# create test set for retrieval settings.
import json
data = json.load(open("data/conala-test.json"))

posts = {}
queries = {}
candidates = {}

query_id = 0
candi_id = 0

for rec in data:
    intent = rec["intent"]
    snippet = rec["snippet"]
    try: posts[intent].append(snippet)
    except KeyError: posts[intent] = [snippet]

for i, rec in enumerate(data):
    intent = rec["intent"]
    snippet = rec["snippet"]
    try:
        queries[intent]+1
    except KeyError:
        queries[intent] = query_id
        query_id += 1
    try:
        candidates[snippet]+1
    except KeyError:
        candidates[snippet] = candi_id
        candi_id += 1
        
query_map = []
for query in queries:
    snippets = posts[query] 
    snippet_ids = [candidates[snippet] for snippet in snippets]
    query_map.append({"query": query, "docs": snippet_ids})

with open("candidate_snippets.json", "w") as f:
    json.dump(
        list(
            candidates.keys()
        ),           
        f, indent=4
    )
with open("query_and_candidates.json", "w") as f:
    json.dump(
        query_map, 
        f, indent=4
    )