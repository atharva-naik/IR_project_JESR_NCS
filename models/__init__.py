# This package contains main model files, for our "Universal Joint/Shared Space Encoder"
# some common utilites.
import os
import json
import torch
import argparse
import numpy as np
from typing import *
from sklearn.metrics import ndcg_score as NDCG
from models.metrics import TripletAccuracy, recall_at_k 
from sklearn.metrics import label_ranking_average_precision_score as MRR

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

def test_ood_performance(triplet_net, model_name: str, query_paths: List[str], 
                         cand_paths: List[str], args: argparse.Namespace,
                         dataset_names: List[str]=["CoNaLa", "External Knowledge"]):
    # do only code retrieval with l2 distance as distance function
    device = args.device_id if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(args.exp_name, "model.pt")
    print(f"loading checkpoint (state dict) from {ckpt_path}")
    try: state_dict = torch.load(ckpt_path, map_location="cpu")
    except Exception as e: 
        state_dict = None
        print("Couldn't load state dict because:")
        print(e)
    # load model checkpoint.
    if state_dict: 
        print(f"\x1b[32;1mloading state dict from {ckpt_path}\x1b[0m")
        triplet_net.load_state_dict(state_dict)
    ID = 0
    all_metrics = {}
    for query_path, cand_path in zip(query_paths, cand_paths):
        # load code candidates.
        print(f"loading candidates from {cand_path}")
        code_and_annotations = json.load(open(cand_path))
        candidates = code_and_annotations["snippets"]
        dataset_name = dataset_names[ID]
        ID += 1
        all_metrics[dataset_name] = {}
        # loading query data.
        print(f"loading queries from {query_path}")
        queries_and_cand_labels = json.load(open(query_path))
        queries = [i["query"] for i in queries_and_cand_labels]
        labels = [i["docs"] for i in queries_and_cand_labels]
        # distance function to be used.
        dist_fn ="l2_dist"
        # assert dist_fn in ["l2_dist", "inner_prod"]
        # encode queries.
        print(f"encoding {len(queries)} queries:")
        query_mat = triplet_net.encode_emb(queries, mode="text", 
                                           batch_size=args.batch_size,
                                           use_tqdm=True, device_id=device)
        query_mat = torch.stack(query_mat)
        # encode candidates.
        print(f"encoding {len(candidates)} candidates:")
        cand_mat = triplet_net.encode_emb(candidates, mode="code", 
                                          batch_size=args.batch_size,
                                          use_tqdm=True, device_id=device)
        # score and rank documents.
        cand_mat = torch.stack(cand_mat)
        scores = torch.cdist(query_mat, cand_mat, p=2)
        doc_ranks = scores.argsort(axis=1)
        # compute recall@k for various k
        recall_at_ = []
        for i in range(1,10+1):
            recall_at_.append(recall_at_k(labels, doc_ranks.tolist(), k=5*i))
        # compute LRAP.
        lrap_GT = np.zeros((len(queries), len(candidates)))
        for i in range(len(labels)):
            for j in labels[i]:
                lrap_GT[i][j] = 1
        # compute micro average and average best label candidate rank
        label_ranks = []
        avg_rank = 0
        avg_best_rank = 0 
        N = 0
        M = 0
        for i, rank_list in enumerate(doc_ranks):
            rank_list = rank_list.tolist()
            # if dist_func == "inner_prod": rank_list = rank_list.tolist()[::-1]
            # elif dist_func == "l2_dist": rank_list = rank_list.tolist()
            instance_label_ranks = []
            ranks = []
            for cand_rank in labels[i]:
                # print(rank_list, cand_rank)
                rank = rank_list.index(cand_rank)
                avg_rank += rank
                ranks.append(rank)
                N += 1
                instance_label_ranks.append(rank)
            M += 1
            avg_best_rank += min(ranks)
            label_ranks.append(instance_label_ranks)
        # compute MRR and NDCG.
        mrr = MRR(lrap_GT, -scores.cpu().numpy())
        ndcg = NDCG(lrap_GT, -scores.cpu().numpy())
        metrics = {
            "avg_candidate_rank": avg_rank/N,
            "avg_best_candidate_rank": avg_best_rank/M,
            "recall": {
                f"@{5*i}": recall_at_[i-1] for i in range(1,10+1) 
            },
        }
        metrics["mrr"] = mrr
        metrics["ndcg"] = ndcg
        all_metrics[dataset_name] = metrics
        # print metrics:
        print("avg canditate rank:", avg_rank/N)
        print("avg best candidate rank:", avg_best_rank/M)
        for i in range(1,10+1):
            print(f"recall@{5*i} = {recall_at_[i-1]}")
        print("NDCG:", ndcg)
        print("MRR (LRAP):", mrr)
    metrics_path = os.path.join(args.exp_name, 
                   "ood_test_metrics_l2_code.json")
    # write metrics to path.
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f)
        
    return all_metrics