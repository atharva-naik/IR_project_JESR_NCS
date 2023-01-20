#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--k", type=int, default=10, help="recall@k used to find the error cases")
# parser.add_argument("-p1", "--preds1", required=True, help="path to predictions of 1st model") # the better model.
# parser.add_argument("-p2", "--preds2", required=True, help="path to predictions of 2nd model") # the worse model.
parser.add_argument("-e1", "--exp1", required=True, help="name of the experiment (1st model)") # the better model.
parser.add_argument("-e2", "--exp2", required=True, help="name of the experiment (2nd model)") # the worse model.
parser.add_argument("-d", "--dataset", type=str, required=True, help="the dataset")
args = parser.parse_args()

# should be in the approved list of datasets.
assert args.dataset in ["CoNaLa", "PyDocs", "WebQuery"]
test_path = {
    "CoNaLa": "query_and_candidates.json", 
    "PyDocs": "external_knowledge/queries.json",
    "WebQuery": "data/queries_webquery.json"}[args.dataset]
canidates_path = {
    "CoNaLa": "candidate_snippets.json", 
    "PyDocs": "external_knowledge/candidates.json",
    "WebQuery": "data/candidates_webquery.json"}[args.dataset]
exp_path_map = {
    "CoNaLa": "CoNaLa Doc Ranks.json",
    "PyDocs": "External Knowledge Doc Ranks.json",
    "WebQuery": "Web Query Doc Ranks.json",
}
exp_path1 = os.path.join("experiments", args.exp1, 
                         exp_path_map[args.dataset])
exp_path2 = os.path.join("experiments", args.exp2, 
                         exp_path_map[args.dataset])
path = os.path.join("improvement_egs", f"{args.exp1}_minus_{args.exp2}_{args.dataset}.json") 
k = args.k
bmgc_wmgw = 0
bmgw_wmgc = 0 

preds1 = json.load(open(exp_path1))
preds2 = json.load(open(exp_path2))
candidates = json.load(open(canidates_path))['snippets']
trues = [i['docs'] for i in json.load(open(test_path))]
queries = [i['query'] for i in json.load(open(test_path))]
query_and_corresponding_labels = []

for pred_row1, pred_row2, labels, query in zip(preds1, preds2, trues, queries):
    worse_model_gets_correct = False
    better_model_gets_correct = False
    for label in labels:
        if label in pred_row1[:k]:
            better_model_gets_correct = True
            break
    for label in labels:
        if label in pred_row2[:k]:
            worse_model_gets_correct = True; break
    
    if better_model_gets_correct and not worse_model_gets_correct:
        codes1 = [candidates[l] for l in pred_row1[:k]]
        codes2 = [candidates[l] for l in pred_row2[:k]]
        true_codes = [candidates[l] for l in labels]
        query_and_corresponding_labels.append((
            query, codes1, codes2, true_codes,
        ))
        bmgc_wmgw += 1
    if not better_model_gets_correct and worse_model_gets_correct:
        bmgw_wmgc += 1
        
print(f"better model gets correct but worse model gets wrong: {bmgc_wmgw}")
print(f"better model gets wrong but worse model gets correct: {bmgw_wmgc}")
with open(path, "w") as f:
    json.dump(query_and_corresponding_labels, f, indent=4)
#
# scripts/find_improvement_egs.py -e1 GraphCodeBERT_ast_5_100k -e2 GraphCodeBERT_100k -m GraphCodeBERT -d PyDocs -k 5
# scripts/find_improvement_egs.py -e1 GraphCodeBERT_ast_5_100k -e2 GraphCodeBERT_100k -m GraphCodeBERT -d WebQuery -k 5