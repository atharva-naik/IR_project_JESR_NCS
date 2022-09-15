#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import matplotlib
import matplotlib.pyplot as plt

def get_stats(exp_name):
    return json.load(open(
        os.path.join(
            "experiments", exp_name,
            "ood_test_metrics_l2_code.json"
        )
    ))
# Datasets:
datasets = ["CoNaLa", "External Knowledge", "Web Query", "CodeSearchNet"]

# baseline variants.

# CodeBERT
codebert = get_stats("CodeBERT_100k")
# GraphCodeBERT
graphcodebert = get_stats("GraphCodeBERT_100k")
# UniXcoder
unixcoder = get_stats("UniXcoder_100k")

# DNS variants.

# CodeBERT + DNS
codebert_dns = get_stats("CodeBERT_dyn_neg_sample_100k")
# GraphCodeBERT + DNS
graphcodebert_dns = get_stats("GraphCodeBERT_dyn_neg_sample_100k")
# UniXcoder + DNS
unixcoder_dns = get_stats("UniXcoder_dyn_neg_sample_100k")

# AST variants.

# CodeBERT + AST
codebert_ast = get_stats("CodeBERT_ast_18_100k")
# GraphCodeBERT + AST
graphcodebert_ast = get_stats("GraphCodeBERT_ast_4_100k")
# UniXcoder + AST
unixcoder_ast = get_stats("UniXcoder_ast_27_100k_warmup_13")

yticks = [r"     MRR", r"     NDCG", r"     R@5", r"CNL: R@10", 
          r"     MRR", r"     NDCG", r"     R@5", r" PD: R@10", 
          r"     MRR", r"     NDCG", r"     R@5", r" WQ: R@10", 
          r"     MRR", r"     NDCG", r"     R@5", r"CSN: R@10"]

# all the metrics.
y_c_mrr = [codebert[dataset]["mrr"] for dataset in datasets]
y_gc_mrr = [graphcodebert[dataset]["mrr"] for dataset in datasets]
y_u_mrr = [unixcoder[dataset]["mrr"] for dataset in datasets[:-1]]

y_c_ndcg = [codebert[dataset]["ndcg"] for dataset in datasets]
y_gc_ndcg = [graphcodebert[dataset]["ndcg"] for dataset in datasets]
y_u_ndcg = [unixcoder[dataset]["ndcg"] for dataset in datasets[:-1]]

y_c_recall5 = [codebert[dataset]["recall"]["@5"] for dataset in datasets]
y_gc_recall5 = [graphcodebert[dataset]["recall"]["@5"] for dataset in datasets]
y_u_recall5 = [unixcoder[dataset]["recall"]["@5"] for dataset in datasets[:-1]]

y_c_recall10 = [codebert[dataset]["recall"]["@10"] for dataset in datasets]
y_gc_recall10 = [graphcodebert[dataset]["recall"]["@10"] for dataset in datasets]
y_u_recall10 = [unixcoder[dataset]["recall"]["@10"] for dataset in datasets[:-1]]

y_c_dns_mrr = [codebert_dns[dataset]["mrr"] for dataset in datasets]
y_gc_dns_mrr = [graphcodebert_dns[dataset]["mrr"] for dataset in datasets]
y_u_dns_mrr = [unixcoder_dns[dataset]["mrr"] for dataset in datasets[:-1]]

y_c_dns_ndcg = [codebert_dns[dataset]["ndcg"] for dataset in datasets]
y_gc_dns_ndcg = [graphcodebert_dns[dataset]["ndcg"] for dataset in datasets]
y_u_dns_ndcg = [unixcoder_dns[dataset]["ndcg"] for dataset in datasets[:-1]]

y_c_dns_recall5 = [codebert_dns[dataset]["recall"]["@5"] for dataset in datasets]
y_gc_dns_recall5 = [graphcodebert_dns[dataset]["recall"]["@5"] for dataset in datasets]
y_u_dns_recall5 = [unixcoder_dns[dataset]["recall"]["@5"] for dataset in datasets[:-1]]

y_c_dns_recall10 = [codebert_dns[dataset]["recall"]["@10"] for dataset in datasets]
y_gc_dns_recall10 = [graphcodebert_dns[dataset]["recall"]["@10"] for dataset in datasets]
y_u_dns_recall10 = [unixcoder_dns[dataset]["recall"]["@10"] for dataset in datasets[:-1]]

y_c_ast_mrr = [codebert_ast[dataset]["mrr"] for dataset in datasets]
y_gc_ast_mrr = [graphcodebert_ast[dataset]["mrr"] for dataset in datasets]
y_u_ast_mrr = [unixcoder_ast[dataset]["mrr"] for dataset in datasets[:-1]]

y_c_ast_ndcg = [codebert_ast[dataset]["ndcg"] for dataset in datasets]
y_gc_ast_ndcg = [graphcodebert_ast[dataset]["ndcg"] for dataset in datasets]
y_u_ast_ndcg = [unixcoder_ast[dataset]["ndcg"] for dataset in datasets[:-1]]

y_c_ast_recall5 = [codebert_ast[dataset]["recall"]["@5"] for dataset in datasets]
y_gc_ast_recall5 = [graphcodebert_ast[dataset]["recall"]["@5"] for dataset in datasets]
y_u_ast_recall5 = [unixcoder_ast[dataset]["recall"]["@5"] for dataset in datasets[:-1]]

y_c_ast_recall10 = [codebert_ast[dataset]["recall"]["@10"] for dataset in datasets]
y_gc_ast_recall10 = [graphcodebert_ast[dataset]["recall"]["@10"] for dataset in datasets]
y_u_ast_recall10 = [unixcoder_ast[dataset]["recall"]["@10"] for dataset in datasets[:-1]]

RED = "#9e3434"
BLUE = "#326ba8"
GREEN = "#459645"
LIGHT_RED = "#f56a5b"
LIGHT_BLUE = "#6bb3e3"
LIGHT_GREEN = "#73d982"
LIGHTER_RED = "#f59e84"
LIGHTER_BLUE = "#94f4ff"
LIGHTER_GREEN = "#97f7ce"

x_recall5 = [0, 2, 4, 6] 
x_recall10 = [1.5, 3.5, 5.5, 7.5]
yticks = ["", "", "CoNaLa", "", 
          "", "", "PyDocs", "", 
          "", "", "", "WebQuery", 
          "", "", "", "Code\nSearch Net"]

plt.figure(figsize=(12,5))
plt.title("""Dataset wise performance breakdown for CodeBERT, GraphCodeBERT, UniXcoder (regular, DNS, AST) recall@(5,10)""", fontsize=12)
plt.scatter(y_c_ast_recall5, x_recall5, s=100, marker="^", facecolors=RED, edgecolors=RED, label="CodeBERT+AST", linewidth=2)
plt.scatter(y_c_ast_recall10, x_recall10, s=100, marker="o", facecolors=RED, edgecolors=RED, linewidth=2)

plt.scatter(y_c_dns_recall5, x_recall5, s=75, marker="^", facecolors='none', edgecolors=LIGHT_RED, label="CodeBERT+DNS", linewidth=2)
plt.scatter(y_c_dns_recall10, x_recall10, s=75, marker="o", facecolors='none', edgecolors=LIGHT_RED, linewidth=2)

plt.scatter(y_c_recall5, x_recall5, s=50, marker="^", facecolors='none', edgecolors=LIGHTER_RED, label="CodeBERT R@5", linewidth=2)
plt.scatter(y_c_recall10, x_recall10, s=50, marker="o", facecolors='none', edgecolors=LIGHTER_RED, label="CodeBERT R@10", linewidth=2)

plt.scatter(y_gc_ast_recall5, x_recall5, s=100, marker="^", facecolors=GREEN, edgecolors=GREEN, label="GraphCodeBER+AST", linewidth=2)
plt.scatter(y_gc_ast_recall10, x_recall10, s=100, marker="o", facecolors=GREEN, edgecolors=GREEN, linewidth=2)

plt.scatter(y_gc_dns_recall5, x_recall5, s=75, marker="^", facecolors='none', edgecolors=LIGHT_GREEN, label="GraphCodeBERT+DNS", linewidth=2)
plt.scatter(y_gc_dns_recall10, x_recall10, s=75, marker="o", facecolors='none', edgecolors=LIGHT_GREEN, linewidth=2)

plt.scatter(y_gc_recall5, x_recall5, s=50, marker="^", facecolors='none', edgecolors=LIGHTER_GREEN, label="GraphCodeBERT", linewidth=2)
plt.scatter(y_gc_recall10, x_recall10, s=50, marker="o", facecolors='none', edgecolors=LIGHTER_GREEN, linewidth=2)

plt.scatter(y_u_ast_recall5, x_recall5[:-1], s=100, marker="^", facecolors=BLUE, edgecolors=BLUE, label="UniXcoder+AST", linewidth=2)
plt.scatter(y_u_ast_recall10, x_recall10[:-1], s=100, marker="o", facecolors=BLUE, edgecolors=BLUE, linewidth=2)

plt.scatter(y_u_dns_recall5, x_recall5[:-1], s=75, marker="^", facecolors='none', edgecolors=LIGHT_BLUE, label="UniXcoder+DNS", linewidth=2)
plt.scatter(y_u_dns_recall10, x_recall10[:-1], s=75, marker="o", facecolors='none', edgecolors=LIGHT_BLUE, linewidth=2)

plt.scatter(y_u_recall5, x_recall5[:-1], s=50, marker="^", facecolors='none', edgecolors=LIGHTER_BLUE, label="UniXcoder", linewidth=2)
plt.scatter(y_u_recall10, x_recall10[:-1], s=50, marker="o", facecolors='none', edgecolors=LIGHTER_BLUE, linewidth=2)


plt.yticks([0, 0.5, 1, 1.5, 
            2, 2.5, 3, 3.5, 
            4, 4.5, 5, 5.5, 
            6, 6.5, 7, 7.5], 
           labels=yticks,
           rotation="90")
x = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
plt.xticks(x, labels=x)
plt.plot([0.4, 1], [1.75, 1.75], color="black", linewidth=0.7)
plt.plot([0.4, 1], [3.75, 3.75], color="black", linewidth=0.7)
plt.plot([0.4, 1], [5.75, 5.75], color="black", linewidth=0.7)

plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
# plt.grid(which='major', color='gray', linestyle='dashed')
plt.savefig("plots/dataset_wise_all_transformers_recalls.png")

plt.clf()
x_mrr = [0, 2, 4, 6] 
x_ndcg = [1.5, 3.5, 5.5, 7.5]
plt.figure(figsize=(12,5))
plt.title("""Dataset wise performance breakdown for CodeBERT, GraphCodeBERT, UniXcoder (regular, DNS, AST) MRR & NDCG""", fontsize=12)
plt.scatter(y_c_ast_mrr, x_mrr, s=100, marker="^", facecolors=RED, edgecolors=RED, label="CodeBERT+AST MRR", linewidth=2)
plt.scatter(y_c_ast_ndcg, x_ndcg, s=100, marker="o", facecolors=RED, edgecolors=RED, label="CodeBERT+AST NDCG", linewidth=2)

plt.scatter(y_c_dns_mrr, x_mrr, s=75, marker="^", facecolors='none', edgecolors=LIGHT_RED, label="CodeBERT+DNS MRR", linewidth=2)
plt.scatter(y_c_dns_ndcg, x_ndcg, s=75, marker="o", facecolors='none', edgecolors=LIGHT_RED, label="CodeBERT+DNS NDCG", linewidth=2)

plt.scatter(y_c_mrr, x_mrr, s=50, marker="^", facecolors='none', edgecolors=LIGHTER_RED, label="CodeBERT MRR", linewidth=2)
plt.scatter(y_c_ndcg, x_ndcg, s=50, marker="o", facecolors='none', edgecolors=LIGHTER_RED, label="CodeBERT NDCG", linewidth=2)

plt.scatter(y_gc_ast_mrr, x_mrr, s=100, marker="^", facecolors=GREEN, edgecolors=GREEN, label="GraphCodeBERT+AST MRR", linewidth=2)
plt.scatter(y_gc_ast_ndcg, x_ndcg, s=100, marker="o", facecolors=GREEN, edgecolors=GREEN, label="GraphCodeBERT+AST NDCG", linewidth=2)

plt.scatter(y_gc_dns_mrr, x_mrr, s=75, marker="^", facecolors='none', edgecolors=LIGHT_GREEN, label="GraphCodeBERT+DNS MRR", linewidth=2)
plt.scatter(y_gc_dns_ndcg, x_ndcg, s=75, marker="o", facecolors='none', edgecolors=LIGHT_GREEN, label="GraphCodeBERT+DNS NDCG", linewidth=2)

plt.scatter(y_gc_mrr, x_mrr, s=50, marker="^", facecolors='none', edgecolors=LIGHTER_GREEN, label="GraphCodeBERT MRR", linewidth=2)
plt.scatter(y_gc_ndcg, x_ndcg, s=50, marker="o", facecolors='none', edgecolors=LIGHTER_GREEN, label="GraphCodeBERT NDCG", linewidth=2)

plt.scatter(y_u_ast_mrr, x_mrr[:-1], s=100,  marker="^", facecolors=BLUE, edgecolors=BLUE, label="UniXcoder+AST MRR", linewidth=2)
plt.scatter(y_u_ast_ndcg, x_ndcg[:-1], s=100, marker="o", facecolors=BLUE, edgecolors=BLUE, label="UniXcoder+AST NDCG", linewidth=2)

plt.scatter(y_u_dns_mrr, x_mrr[:-1], s=75,  marker="^", facecolors='none', edgecolors=LIGHT_BLUE, label="UniXcoder+DNS MRR", linewidth=2)
plt.scatter(y_u_dns_ndcg, x_ndcg[:-1], s=75, marker="o", facecolors='none', edgecolors=LIGHT_BLUE, label="UniXcoder+DNS NDCG", linewidth=2)

plt.scatter(y_u_mrr, x_mrr[:-1], s=50, marker="^", facecolors='none', edgecolors=LIGHTER_BLUE, label="UniXcoder MRR", linewidth=2)
plt.scatter(y_u_ndcg, x_ndcg[:-1], s=50, marker="o", facecolors='none', edgecolors=LIGHTER_BLUE, label="UniXcoder NDCG", linewidth=2)


plt.yticks([0, 0.5, 1, 1.5, 
            2, 2.5, 3, 3.5, 
            4, 4.5, 5, 5.5, 
            6, 6.5, 7, 7.5], 
           labels=yticks,
           rotation="90")
x = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
plt.xticks(x, labels=x)
plt.plot([0.3, 1], [1.75, 1.75], color="black", linewidth=0.7)
plt.plot([0.3, 1], [3.75, 3.75], color="black", linewidth=0.7)
plt.plot([0.3, 1], [5.75, 5.75], color="black", linewidth=0.7)

plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
# plt.grid(which='major', color='gray', linestyle='dashed')
plt.savefig("plots/dataset_wise_all_transformers_mrr_ndcg.png")