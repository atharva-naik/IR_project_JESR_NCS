import os
import json
import torch
import argparse
import numpy as np
from typing import *
import torch.nn as nn
from sklearn.metrics import ndcg_score as NDCG
from sklearn.metrics import label_ranking_average_precision_score as MRR

# compute recall@k.
def recall_at_k(actual, predicted, k: int=10):
    rel = 0
    tot = 0
    for act_list, pred_list in zip(actual, predicted):
        for i in act_list:
            tot += 1
            if i in pred_list[:k]: rel += 1
                
    return rel/tot
# do out of domain performance testing.
def test_ood_performance(encoder, query_paths: List[str], cand_paths: List[str], 
                         args: argparse.Namespace, dataset_names: List[str]=[
                             "CoNaLa", "External Knowledge", 
                             "Web Query", "CodeSearchNet",
                         ]):
    # do only code retrieval with l2 distance as distance function
    device = args.device_id if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(args.exp_name, "model.pt")
    print(f"loading checkpoint (state dict) from {ckpt_path}")
    try: state_dict = torch.load(ckpt_path, map_location="cpu")
    except Exception as e: 
        state_dict = None
        print(f"Couldn't load state dict because: {e}")
    if state_dict: # load model checkpoint.
        print(f"\x1b[32;1mloading state dict from {ckpt_path}\x1b[0m")
        encoder.load_state_dict(state_dict)
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
        query_mat = encoder.encode(queries, mode="text", 
                                   device_id=device, enc_type=args.enc_type,
                                   use_tqdm=True, batch_size=args.batch_size)
        query_mat = torch.stack(query_mat)
        # encode candidates.
        print(f"encoding {len(candidates)} candidates:")
        cand_mat = encoder.encode(candidates, mode="code", 
                                  device_id=device, enc_type=args.enc_type,
                                  use_tqdm=True, batch_size=args.batch_size)
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
        N, M = 0, 0
        label_ranks = []
        avg_rank, avg_best_rank = 0, 0
        for i, rank_list in enumerate(doc_ranks):
            rank_list = rank_list.tolist()
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