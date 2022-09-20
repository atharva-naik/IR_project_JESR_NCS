#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from typing import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from models.CodeBERT import CodeBERTripletNet
from models.UniXcoder import UniXcoderTripletNet
from models.GraphCodeBERT import GraphCodeBERTripletNet

parser = argparse.ArgumentParser()
parser.add_argument("-ea", "--exp_AST", required=True, 
                    help="name of the experiment (1st model)")
parser.add_argument("-eb", "--exp_base", required=True, 
                    help="name of the experiment (2nd model)")
parser.add_argument("-d", "--dataset", type=str, 
                    required=True, help="the dataset")
parser.add_argument("-m", "--model_type", type=str, 
                    required=True, help="the type of the model")
parser.add_argument("-id", "--device_id", type=str, 
                    default="cuda:0", help="GPU device ID to be used")
args = parser.parse_args()

# save the embeddings of text (query) and code (candidates) to the experiment folder.

# hardcoded examples (query IDs) from improvement egs. (change according to the model being used)
if args.model_type == "codebert":
    if args.dataset == "CoNaLa":
        query_IDs = [4, 19, 22, 17, 36]

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
train_set = {
    "CoNaLa": "data/conala-mined-100k_train.json",
}
val_set = {
    "CoNaLa": "data/conala-mined-100k_val.json"
}
# load trainset
# get path to tokenizer.
def get_tok_path(model_name: str) -> str:
    assert model_name in ["codebert", "graphcodebert", "unixcoder"]
    if model_name == "codebert":
        tok_path = os.path.expanduser("~/codebert-base-tok")
        if not os.path.exists(tok_path):
            tok_path = "microsoft/codebert-base"
    elif model_name == "graphcodebert":
        tok_path = os.path.expanduser("~/graphcodebert-base-tok")
        if not os.path.exists(tok_path):
            tok_path = "microsoft/grapcodebert-base"
    elif model_name == "unixcoder":
        tok_path = os.path.expanduser("~/unixcoder-base-tok")
        if not os.path.exists(tok_path):
            tok_path = "microsoft/unixcoder-base"
            
    return tok_path
# load candidates, queries and gold candidate IDs.
candidates = json.load(open(canidates_path))['snippets']
queries = [i['query'] for i in json.load(open(test_path))]
true_cands = [i['docs'] for i in json.load(open(test_path))]

AST_code_emb_save_path = os.path.join(args.exp_AST, "cand_embs.pt")
AST_nl_emb_save_path = os.path.join(args.exp_AST, "query_embs.pt")
base_code_emb_save_path = os.path.join(args.exp_base, "cand_embs.pt")
base_nl_emb_save_path = os.path.join(args.exp_base, "query_embs.pt")
# check if AST or baseline model code/queries need to be encoded.
encode_AST_egs = not(os.path.exists(AST_code_emb_save_path) and os.path.exists(AST_nl_emb_save_path))
encode_base_egs = not(os.path.exists(base_code_emb_save_path) and os.path.exists(base_nl_emb_save_path))

# if candidate or query embedding vectors exist or not:
if encode_AST_egs or encode_base_egs:
    # create model object and load checkpoint.
    tok_path = get_tok_path(args.model_type)
    if args.model_type == "unixcoder":
        model = UniXcoderTripletNet()
    elif args.model_type == "codebert":
        model = CodeBERTripletNet(tok_path=tok_path)
    elif args.model_type == "graphcodebert":
        model = GraphCodeBERTripletNet(tok_path=tok_path)
    
if encode_AST_egs:
    model.load_state_dict(torch.load(os.path.join(
        args.exp_AST, "model.pt"), 
        map_location="cpu"
    ))
    print(f"loaded state_dict from: {os.path.join(args.exp_AST, 'model.pt')}")
    # encode candidate and query matrices.
    cand_mat_AST = model.encode_emb(candidates, mode="code", batch_size=48, 
                                    use_tqdm=True, device=args.device_id)
    query_mat_AST = model.encode_emb(queries, mode="text", batch_size=48, 
                                     use_tqdm=True, device=args.device_id)
    # stack the list of tensors to get 2D tensor:
    cand_mat_AST = torch.stack(cand_mat_AST)
    query_mat_AST = torch.stack(query_mat_AST)
    print(cand_mat_AST.shape)
    print(query_mat_AST.shape)
    # save the candidates and queries to .pt files.
    print(f"saving candidate embeddings for AST to: {AST_code_emb_save_path}")
    torch.save(cand_mat_AST, AST_code_emb_save_path)
    print(f"saving query embeddings for AST to: {AST_nl_emb_save_path}")
    torch.save(query_mat_AST, AST_nl_emb_save_path)
else:
    # load candidate and query embeddings.
    print(f"loading candidate embeddings from: {AST_code_emb_save_path}")
    cand_mat_AST = torch.load(AST_code_emb_save_path, map_location="cpu")
    print(f"loading query embeddings from: {AST_nl_emb_save_path}")
    query_mat_AST = torch.load(AST_nl_emb_save_path, map_location="cpu")
    # verify if the candidate & query matrix shapes are correct.
    print(cand_mat_AST.shape)
    print(query_mat_AST.shape)
    
if encode_base_egs:                              
    model.load_state_dict(torch.load(os.path.join(
        args.exp_base, "model.pt"), 
        map_location="cpu"
    ))
    print(f"loaded state_dict from: {os.path.join(args.exp_base, 'model.pt')}")
    # encode candidate and query matrices.
    cand_mat_base = model.encode_emb(candidates, mode="code", batch_size=48, 
                                     use_tqdm=True, device=args.device_id)
    query_mat_base = model.encode_emb(queries, mode="text", batch_size=48, 
                                      use_tqdm=True, device=args.device_id)
    # stack the list of tensors to get 2D tensor:
    cand_mat_base = torch.stack(cand_mat_base)
    query_mat_base = torch.stack(query_mat_base)
    print(cand_mat_base.shape)
    print(query_mat_base.shape)
    # save the candidates and queries to .pt files.
    print(f"saving candidate embeddings for baseline to: {base_code_emb_save_path}")
    torch.save(cand_mat_base, base_code_emb_save_path)
    print(f"saving query embeddings for baseline to: {base_nl_emb_save_path}")
    torch.save(query_mat_base, base_nl_emb_save_path)
else:
    # load candidate and query embeddings.
    print(f"loading candidate embeddings from: {base_code_emb_save_path}")
    cand_mat_base = torch.load(base_code_emb_save_path, map_location="cpu")
    print(f"loading query embeddings from: {base_nl_emb_save_path}")
    query_mat_base = torch.load(base_nl_emb_save_path, map_location="cpu")
    # verify if the candidate & query matrix shapes are correct.
    print(cand_mat_base.shape)
    print(query_mat_base.shape)
    
# load the model predictions (from the experiment folder):
preds_AST = json.load(open(os.path.join(
    args.exp_AST, "CoNaLa Doc Ranks.json",
)))
preds_base = json.load(open(os.path.join(
    args.exp_base, "CoNaLa Doc Ranks.json",
)))
cand_mat_AST = cand_mat_AST.tolist()
query_mat_AST = query_mat_AST.tolist()
cand_mat_base = cand_mat_base.tolist()
query_mat_base = query_mat_base.tolist()
# fit tsne transform.
cand_sep = len(cand_mat_AST)
tsne = TSNE(n_components=2, verbose=1, random_state=42)
z_AST = tsne.fit_transform(cand_mat_AST+query_mat_AST)
tsne = TSNE(n_components=2, verbose=1, random_state=42)
z_base = tsne.fit_transform(cand_mat_base+query_mat_base)

# def looks_better_dist(q, c): # for the 2:1 plot dimensions. 
#     return np.sqrt(4*(q[0]-c[0])**2 + (q[1]-c[1])**2)
# optimal selection of query IDs.
# # determine the query_IDs which look best on the t-SNE plot:
# scores = []
# for i in range(len(queries)): 
#     dist_margins = []
#     for j in true_cands[i]:
#         q_vec = z_AST[cand_sep:][i]
#         tc_vec = z_AST[:cand_sep][i]
#         AST_dist = looks_better_dist(q_vec, tc_vec)
#         q_vec = z_base[cand_sep:][i]
#         tc_vec = z_base[:cand_sep][i]
#         base_dist = looks_better_dist(q_vec, tc_vec)
#         # print(f"AST_dist: {AST_dist} base_dist: {base_dist}")
#         dist_margins.append(base_dist-AST_dist)
#     scores.append(np.mean(dist_margins))
# print(np.argsort(scores)[:5])
# query_IDs = np.argsort(scores)[:5].tolist()
# print(f"average dist-margin: {np.mean(dist_margins)}")

# find query and candidates IDs.
AST_cand_IDs = {}
base_cand_IDs = {}
true_cand_IDs = {}
for id in query_IDs:
    # the top-k k used (hard coded to just 5 for now)
    AST_cand_IDs[id] = preds_AST[id][:5]
    base_cand_IDs[id] = preds_base[id][:5]
    true_cand_IDs[id] = true_cands[id]
# query IDs, AST@5 candidate IDs, baseline@5 candidate IDs, true candidate IDs for queries.
print(f"query_IDs: {query_IDs}")
print(f"AST@5 cand_IDs: {AST_cand_IDs}")
print(f"base@5 cand_IDs: {base_cand_IDs}")
print(f"true_cand_IDs: {true_cand_IDs}")
# first index controls model: AST/baseline
# second (dictionary) index: query ID
# third index (in list value for a dictionary key): candidate rank
# value: float value of component-1/component-2
code_comp1: List[Dict[int, List[int]]] = [{}, {}]  
code_comp2: List[Dict[int, List[int]]] = [{}, {}]
true_comp1: List[Dict[int, List[int]]] = [{}, {}]
true_comp2: List[Dict[int, List[int]]] = [{}, {}]
# first index controls model: AST/baseline
# second (dictionary) index: query ID
# value: float value of component-1/component-2
nl_comp1: List[Dict[int, int]] = [{}, {}]
nl_comp2: List[Dict[int, int]] = [{}, {}]
for i in query_IDs: 
    # NL comp-1/comp-2
    nl_comp1[0][i] = z_AST[cand_sep:,0][i]
    nl_comp2[0][i] = z_AST[cand_sep:,1][i]
    nl_comp1[1][i] = z_base[cand_sep:,0][i]
    nl_comp2[1][i] = z_base[cand_sep:,1][i]
    # code comp-1/comp-2
    code_comp1[0][i] = []
    code_comp2[0][i] = []
    code_comp1[1][i] = []
    code_comp2[1][i] = []
    # true code candidate comp-1/comp-2
    true_comp1[0][i] = []
    true_comp2[0][i] = []
    true_comp1[1][i] = []
    true_comp2[1][i] = []
    # 
    for j in AST_cand_IDs[i]:
        code_comp1[0][i].append(z_AST[:cand_sep,0][j])
        code_comp2[0][i].append(z_AST[:cand_sep,1][j])
    for j in base_cand_IDs[i]:
        code_comp1[1][i].append(z_base[:cand_sep,0][j])
        code_comp2[1][i].append(z_base[:cand_sep,1][j])
    for j in true_cand_IDs[i]:
        true_comp1[0][i].append(z_AST[:cand_sep,0][j])
        true_comp2[0][i].append(z_AST[:cand_sep,1][j])
        true_comp1[1][i].append(z_base[:cand_sep,0][j])
        true_comp2[1][i].append(z_base[:cand_sep,1][j])
# check code and NL components out:    
print(code_comp1)
print(code_comp2)
print(true_comp1)
print(true_comp2)
print(nl_comp1)
print(nl_comp2)

fig, ax = plt.subplots(1, 1, sharey=True)
# set dimensions of plot.
fig.set_figheight(5)
fig.set_figwidth(10)
# plot different models:
legend_elements = []
markers = ["o","^","*","s", "h", "D", "p", "P"]
model_names = ["AST", "base"]
colors = [["#52ff60", "#ff584d", "#5280ff"], ["#b3ffb9", "#f2aaa5", "#aabdf2"]]
vec_types = ["query", "retrieved@5", "true cands"]
for i in range(2):
    for j in range(len(colors[i])):
        label = f"{model_names[i]}: {vec_types[j]}"
        legend_elements.append(Line2D(
            [0], [0], marker="o", markersize=8,
            markerfacecolor=colors[i][j], 
            color="none", label=label, 
            markeredgewidth=0,
        ))
for i in range(2):
    line_data = []
    # plot different queries:
    for ind, j in enumerate(query_IDs): # green color
        # print(f"nl_comp1[{i}][{j}] = {nl_comp1[i][j]}")
        # print(f"nl_comp2[{i}][{j}] = {nl_comp2[i][j]}")
        ax.scatter(nl_comp1[i][j], nl_comp2[i][j], s=200, 
                        label="query", marker=markers[ind], 
                        edgecolors=colors[i][0], facecolors="none",
                        linewidth=2)
        # plot retrieved code candidates:
        x, y = [], []
        for k in range(5): # red color
            x.append(code_comp1[i][j][k])
            y.append(code_comp2[i][j][k])
            # print(f"code_comp1[{i}][{j}][{k}] = {code_comp1[i][j][k]}")
            # print(f"code_comp2[{i}][{j}][{k}] = {code_comp2[i][j][k]}")
        ax.scatter(x, y, label="retrieved", linewidth=2, s=50,
                   marker=markers[ind], edgecolors=colors[i][1], 
                   facecolors="none")
        # plot true code candidates:
        x, y = [], []
        for k in range(len(true_cand_IDs[j])): # blue color
            x.append(true_comp1[i][j][k])
            y.append(true_comp2[i][j][k])
            line_data.append((nl_comp1[i][j], nl_comp2[i][j]))
            line_data.append((true_comp1[i][j][k], true_comp2[i][j][k]))
            line_data.append("black")
            ax.plot([nl_comp1[i][j], true_comp1[i][j][k]], 
                    [nl_comp2[i][j], true_comp2[i][j][k]],
                    color="black", linestyle="--")
            # print(f"true_comp1[{i}][{j}][{k}] = {true_comp1[i][j][k]}")
            # print(f"true_comp2[{i}][{j}][{k}] = {true_comp2[i][j][k]}")
        ax.scatter(x, y, edgecolors=colors[i][2], facecolors="none",
                   marker=markers[ind], label="true", s=100, linewidth=2)
    # metadata of plot.
    ax.set_xlabel("comp-1")
    ax.set_ylabel("comp-2")
# save plot.
filename = f"tsne_{args.model_type}_{args.dataset.lower()}.png"
ax.legend(handles=legend_elements, loc="upper right")
plot_save_path = os.path.join("plots", filename)
plt.tight_layout()
plt.savefig(plot_save_path)