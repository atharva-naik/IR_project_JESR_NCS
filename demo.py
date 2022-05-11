import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Tuple, List, Dict
from models.GraphCodeBERT import GraphCodeBERTripletNet 

def get_args():
    parser = argparse.ArgumentParser("script to demo annotated code retreival system based on GraphCodeBERT's Late Fusion configuration")
    parser.add_argument("-cp", "--ckpt_path", type=str, default="GraphCodeBERT_rel_thresh/model.pt", help="path to the checkpoint to be loaded")
    parser.add_argument("-uco", "--use_code_only", action="store_true", help="use code only mode")
    parser.add_argument("-c", "--candidates_path", type=str, default="candidate_snippets.json")
    parser.add_argument("-q", "--queries_path", type=str, default="query_and_candidates.json")
    parser.add_argument("-demo", "--demo", action="store_true", help="open in demo mode")
    parser.add_argument("-d", "--device_id", type=str, default="cuda:0")
    parser.add_argument("-df", "--dist_fn", type=str, default="l2_dist") 
    parser.add_argument("-i", "--interactive_mode", action="store_true")
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-k", "--k", type=int, default=5)

    return parser.parse_args()

def get_top_k_results(model: GraphCodeBERTripletNet, queries: List[str], labels: List[list], 
                      cand_mat_code: torch.Tensor, cand_mat_annot: torch.Tensor,
                      args: argparse.Namespace) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """
    takes the encoder model, list of queries, labels (list of list of indices of correct documents)
    candidate matrices (encoded) for code and corresponding annotations & additional arguments (Namespace)
    Returns a dictionary of the k best and worst instances (based on instance level average reciprocal rank)
    or retrieved responses as compared to the labels.
    """
    dist_fn = args.dist_fn
    device = args.device_id if torch.cuda.is_available() else "cpu"
    # create the query matrix for all queries in the test set.
    query_mat = model.encode_emb(queries, mode="text", use_tqdm=False, 
                                 device_id=device, batch_size=1)
    query_mat = torch.stack(query_mat)
    # utilize distance function to score all candidates
    if dist_fn == "inner_prod":
        scores_code = query_mat @ cand_mat_code.T
        scores_annot = query_mat @ cand_mat_annot.T
        scores = scores_code + scores_annot    
    elif dist_fn == "l2_dist":
        scores_code = torch.cdist(query_mat, cand_mat_code, p=2)
        scores_annot = torch.cdist(query_mat, cand_mat_annot, p=2)
        scores = scores_code + scores_annot
    # calculate document ranks.
    ranked_docs = scores.argsort(axis=1)
    if dist_fn == "inner_prod":
        ranked_docs = ranked_docs.flip(dims=[1])
    ranked_docs = ranked_docs.cpu()
    # hard coded to top 5 results.
    return ranked_docs[0][:5]
    
def show_best_worst_examples(model: GraphCodeBERTripletNet, queries: List[str], labels: List[list], 
                             cand_mat_code: torch.Tensor, cand_mat_annot: torch.Tensor,
                             args: argparse.Namespace) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """
    takes the encoder model, list of queries, labels (list of list of indices of correct documents)
    candidate matrices (encoded) for code and corresponding annotations & additional arguments (Namespace)
    Returns a dictionary of the k best and worst instances (based on instance level average reciprocal rank)
    or retrieved responses as compared to the labels.
    """
    k: int=args.k
    # print(f"k={k}")
    print(f"encoding all {len(queries)} queries")
    dist_fn = args.dist_fn
    device = args.device_id if torch.cuda.is_available() else "cpu"
    
    # create the query matrix for all queries in the test set.
    query_mat = model.encode_emb(queries, mode="text", 
                                 use_tqdm=True, device_id=device,
                                 batch_size=args.batch_size)
    query_mat = torch.stack(query_mat)
    # utilize distance function to score all candidates
    if dist_fn == "inner_prod":
        scores_code = query_mat @ cand_mat_code.T
        scores_annot = query_mat @ cand_mat_annot.T
        scores = scores_code + scores_annot    
    elif dist_fn == "l2_dist":
        scores_code = torch.cdist(query_mat, cand_mat_code, p=2)
        scores_annot = torch.cdist(query_mat, cand_mat_annot, p=2)
        scores = scores_code + scores_annot
    # calculate document ranks.
    ranked_docs = scores.argsort(axis=1)
    if dist_fn == "inner_prod":
        ranked_docs = ranked_docs.flip(dims=[1])
    ranked_docs = ranked_docs.cpu()
    # score each instance by how successfully the model classifies it
    # to get the k best and worst examples.
    # use instance averaged (average across no. of candidates per instance) reciprocal ranks.
    instance_rrs = []
    for i, rank_list in enumerate(ranked_docs):
        rank_list = rank_list.tolist()
        rr_sum = 0
        for cand_rank in labels[i]:
            rank = 1+rank_list.index(cand_rank) # index of 0 == rank of 1
            rr_sum += (1/rank)
        instance_rrs.append(rr_sum / len(labels[i]))
    
    # rank instances by instance averaged RR.
    sorted_instance_idx = np.array(instance_rrs).argsort()
    best_query_ids = sorted_instance_idx[::-1][:k]
    worst_query_ids = sorted_instance_idx[:k]
    # final results (best & worst examples).
    best_results = {}
    worst_results = {}
    best_result_GTs = []
    worst_result_GTs = []
    # hard coded: show top-5 results for the query.
    for i in best_query_ids:
        best_results[queries[i]] = []
        best_result_GTs.append(labels[i])
        for j in ranked_docs[i][:5]:
            is_true = int(j in labels[i])
            best_results[queries[i]].append((
                j.item(), 
                is_true
            ))
    # hard coded: show top-5 results for the query.
    for i in worst_query_ids:
        worst_results[queries[i]] = []
        worst_result_GTs.append(labels[i])
        for j in ranked_docs[i][:5]:
            is_true = int(j in labels[i])
            worst_results[queries[i]].append((
                j.item(), 
                is_true
            ))
    
    return best_results, best_result_GTs, worst_results, worst_result_GTs
        
def demo_annotated_code_search(args):
    print("initializing model and tokenizer ..")
    tok_path = os.path.join(
        os.path.expanduser("~"), 
        "graphcodebert-base-tok"
    )
    if not os.path.exists(tok_path):
        tok_path = "graphcodebert-base-tok"
    
    # load state dict from checkpoint path.
    ckpt_path = args.ckpt_path
    print(f"loading checkpoint (state dict) from {ckpt_path}")
    try: state_dict = torch.load(ckpt_path)
    except Exception as e: 
        state_dict = None; print(e)
    # device.
    device = args.device_id if torch.cuda.is_available() else "cpu"
    
    # create model object.
    print("creating model object")
    triplet_net = GraphCodeBERTripletNet(tok_path=tok_path)
    if state_dict: triplet_net.load_state_dict(state_dict)
    print(f"loading candidates from {args.candidates_path}")
    code_and_annotations = json.load(open(args.candidates_path))
    
    # load code and annotation candidates.
    code_candidates = code_and_annotations["snippets"]
    annot_candidates = code_and_annotations["annotations"]
    print(f"loading queries from {args.queries_path}")
    
    # load queries and correct document indices per query.
    queries_and_cand_labels = json.load(open(args.queries_path))
    queries = [i["query"] for i in queries_and_cand_labels]
    labels = [i["docs"] for i in queries_and_cand_labels]
    
    # distance measure to be used.
    dist_func = args.dist_fn
    
    # create the code and annotation candidate matrices. (can be considered an offline process)
    print(f"encoding all {len(code_candidates)} candidates from the CoNaLa test set")
    cand_mat_code = triplet_net.encode_emb(code_candidates, mode="code", 
                                           use_tqdm=True, device_id=device,
                                           batch_size=args.batch_size)
    cand_mat_annot = triplet_net.encode_emb(annot_candidates, mode="text", 
                                            use_tqdm=True, device_id=device,
                                            batch_size=args.batch_size)
    cand_mat_code = torch.stack(cand_mat_code)
    cand_mat_annot = torch.stack(cand_mat_annot)
    
    # interactive mode.
    if args.interactive_mode:
        while True:
            query = input("Enter a natural language query: ").strip()
            if query == ":exit" or query == ":quit":
                print("\x1b[31;1mshutting down!\x1b[0m")
                exit()
            results = get_top_k_results(model=triplet_net, queries=[query],  
                                        cand_mat_annot=cand_mat_annot, 
                                        cand_mat_code=cand_mat_code,
                                        labels=labels, args=args)
            print("\n\x1b[33;1mThe top 5 results are:\x1b[0m\n")
            for i in results:
                code = code_candidates[i]
                annot = annot_candidates[i]
                print(f"{i}) \x1b[34;1m{annot}\x1b[0m\n{code}")    
    # show k of the best and worst examples
    else:
        best_eg, best_GTs, worst_eg, worst_GTs = show_best_worst_examples(model=triplet_net, queries=queries,  
                                                                          cand_mat_annot=cand_mat_annot, 
                                                                          cand_mat_code=cand_mat_code,
                                                                          labels=labels, args=args)
        k = 0
        print("\x1b[32;1mcorrectly\x1b[0m classified examples:")
        for eg, id_list in best_eg.items():
            print(f"\n\x1b[34;1m{eg}\x1b[0m")
            print("\nGround Truths: ")
            print([code_candidates[t] for t in best_GTs[k]])
            print()
            for i, is_true in id_list:
                code = code_candidates[i]
                annot = annot_candidates[i]
                if is_true: print(f"{i}) \x1b[32;1mannot: {annot}\ncode: {code}\x1b[0m")
                else: print(f"{i}) {annot}\n{code}")
            k += 1
        
        k = 0
        print("\x1b[31;1mincorrectly\x1b[0m classified examples:")
        for eg, id_list in worst_eg.items():
            print(f"\n\x1b[34;1m{eg}\x1b[0m")
            print("\nGround Truths: ")
            print([code_candidates[t] for t in worst_GTs[k]])
            print()
            for i, is_true in id_list:
                code = code_candidates[i]
                annot = annot_candidates[i]
                if is_true: print(f"{i}) \x1b[32;1mannot: {annot}\ncode: {code}\x1b[0m")
                else: print(f"{i}) {annot}\n{code}")
            k += 1
        
def save_predictions(args):
    print("initializing model and tokenizer ..")
    tok_path = os.path.join(
        os.path.expanduser("~"), 
        "graphcodebert-base-tok"
    )
    if not os.path.exists(tok_path):
        tok_path = "graphcodebert-base-tok"
    
    # load state dict from checkpoint path.
    ckpt_path = args.ckpt_path
    print(f"loading checkpoint (state dict) from {ckpt_path}")
    try: state_dict = torch.load(ckpt_path)
    except Exception as e: 
        state_dict = None; print(e)
    # device.
    device = args.device_id if torch.cuda.is_available() else "cpu"
    
    # create model object.
    print("creating model object")
    triplet_net = GraphCodeBERTripletNet(tok_path=tok_path)
    if state_dict: triplet_net.load_state_dict(state_dict)
    print(f"loading candidates from {args.candidates_path}")
    code_and_annotations = json.load(open(args.candidates_path))
    
    # load code and annotation candidates.
    code_candidates = code_and_annotations["snippets"]
    annot_candidates = code_and_annotations["annotations"]
    print(f"loading queries from {args.queries_path}")
    
    # load queries and correct document indices per query.
    queries_and_cand_labels = json.load(open(args.queries_path))
    queries = [i["query"] for i in queries_and_cand_labels]
    labels = [i["docs"] for i in queries_and_cand_labels]
    # create the code and annotation candidate matrices. (can be considered an offline process)
    print(f"encoding all {len(code_candidates)} candidates from the CoNaLa test set")
    cand_mat_code = triplet_net.encode_emb(code_candidates, mode="code", 
                                           use_tqdm=True, device_id=device,
                                           batch_size=args.batch_size)
    cand_mat_annot = triplet_net.encode_emb(annot_candidates, mode="text", 
                                            use_tqdm=True, device_id=device,
                                            batch_size=args.batch_size)
    query_mat = triplet_net.encode_emb(queries, mode="text", 
                                       use_tqdm=True, device_id=device,
                                       batch_size=args.batch_size)
    cand_mat_code = torch.stack(cand_mat_code)
    cand_mat_annot = torch.stack(cand_mat_annot)
    query_mat = torch.stack(query_mat)
    # utilize distance function to score all candidates
    if args.dist_fn == "inner_prod":
        scores_code = query_mat @ cand_mat_code.T
        scores_annot = query_mat @ cand_mat_annot.T
    elif args.dist_fn == "l2_dist":
        scores_code = torch.cdist(query_mat, cand_mat_code, p=2)
        scores_annot = torch.cdist(query_mat, cand_mat_annot, p=2)
    # choice between using code only & using both code and annotations.
    if args.use_code_only: scores = scores_code
    else: scores = scores_code + scores_annot
    # calculate document ranks.
    ranked_docs = scores.argsort(axis=1)
    if args.dist_fn == "inner_prod":
        ranked_docs = ranked_docs.flip(dims=[1])
    # the final ranked documents.
    ranked_docs = ranked_docs.cpu().tolist()
    # the top-k ranked docs as retrieved by the model will be stored for each query.
    retrieval_egs = []
    failure_egs = []
    # calculate success stats of model.
    id = 0
    tot = 0
    avg_hits = 0
    recall_at_k = 0
    percent_at_least_1_hit = 0
    # evaluate the model predictions.
    for ranked_doc_list, query, label_ids in zip(ranked_docs, queries, labels):
        label_code_candidates = [code_candidates[i] for i in label_ids]
        label_code_candidates = " <EOC> ".join(label_code_candidates)
        num_cands_in_top_k = 0 # the number of candidates present in the top-k docs retrived by model.
        for i in label_ids:
            tot += 1
            if i in ranked_doc_list[:args.k]:
                recall_at_k += 1
                num_cands_in_top_k += 1
        record = {
            "id": id,
            "query": query,
            f"hits@{args.k}": num_cands_in_top_k,
            "label candidates": label_code_candidates,
        }
        any_hits = (1 if num_cands_in_top_k > 0 else 0)
        avg_hits += num_cands_in_top_k
        percent_at_least_1_hit += any_hits
        if any_hits == 0: failure_egs.append(record)
        for i, ii in enumerate(ranked_doc_list[:args.k]):
            record[f"code@{i+1}"] = code_candidates[ii]
        retrieval_egs.append(record)
        id += 1
    # calc stats.
    N = len(queries)
    avg_hits /= N
    recall_at_k /= tot
    percent_at_least_1_hit /= N
    print(f"average hits: {avg_hits:.3f}")
    print(f"recall@{args.k}: {100*recall_at_k:.3f}%")
    print(f"at least 1 hit for {100*percent_at_least_1_hit:.3f} % queries")
    # the save path is named according to the model checkpoint that was used.
    ckpt_name = Path(args.ckpt_path).parent.name
    eg_save_path = f"retrieval_egs_{ckpt_name}.csv"
    failure_path = f"failure_egs_{ckpt_name}.csv"
    # save the eg. df as a csv file at `eg_save_path`.
    eg_df = pd.DataFrame(retrieval_egs)
    eg_df = eg_df.sort_values(by=f"hits@{args.k}")
    eg_df.to_csv(eg_save_path, index=False)
    # save the failure egs. df as csv file at `failure_path`.
    fail_df = pd.DataFrame(failure_egs)
    fail_df = fail_df.sort_values(by=f"hits@{args.k}")
    fail_df.to_csv(failure_path, index=False)
    
    
if __name__ == "__main__":
    args = get_args()
    if args.demo:
        demo_annotated_code_search(args)
    else:
        save_predictions(args)