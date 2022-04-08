import os
import json
import torch
import argparse
import numpy as np
from typing import Union, Tuple, List, Dict
from models.GraphCodeBERT import GraphCodeBERTripletNet 

def get_args():
    parser = argparse.ArgumentParser("script to demo annotated code retreival system based on GraphCodeBERT's Late Fusion configuration")
    parser.add_argument("-c", "--candidates_path", type=str, default="candidate_snippets.json")
    parser.add_argument("-q", "--queries_path", type=str, default="query_and_candidates.json")
    parser.add_argument("-cp", "--ckpt_path", type=str, default="GraphCodeBERT_rel_thresh/model.pt")
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
    # hard coded: show top-5 results for the query.
    for i in best_query_ids:
        best_results[queries[i]] = []
        for j in ranked_docs[i][:5]:
            is_true = int(j in labels[i])
            best_results[queries[i]].append((
                j.item(), 
                is_true
            ))
    # hard coded: show top-5 results for the query.
    for i in worst_query_ids:
        worst_results[queries[i]] = []
        for j in ranked_docs[i][:5]:
            is_true = int(j in labels[i])
            worst_results[queries[i]].append((
                j.item(), 
                is_true
            ))
    
    return best_results, worst_results
        
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
            if query == ":exit" or query == ":quit":
                print("\x1b[31;1mshutting down!\x1b[0m")
                exit()
            query = input("Enter a natural language query: ").strip()
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
        best_eg, worst_eg = show_best_worst_examples(model=triplet_net, queries=queries,  
                                                     cand_mat_annot=cand_mat_annot, 
                                                     cand_mat_code=cand_mat_code,
                                                     labels=labels, args=args)
        print(best_eg)
        print(worst_eg)
        print("\x1b[32;1mcorrectly\x1b[0m classified examples:")
        for eg, id_list in best_eg.items():
            print(f"\x1b[34;1m{eg}\x1b[0m")
            for i, is_true in id_list:
                code = code_candidates[i]
                annot = annot_candidates[i]
                if is_true: print(f"{i}) \x1b[32;1mannot: {annot}\ncode:{code}\x1b[0m")
                else: print(f"{i}) {annot}\n{code}")
        print("\x1b[31;1mincorrectly\x1b[0m classified examples:")
        for eg, id_list in worst_eg.items():
            print(f"\x1b[34;1m{eg}\x1b[0m")
            for i, is_true in id_list:
                code = code_candidates[i]
                annot = annot_candidates[i]
                if is_true: print(f"{i}) \x1b[32;1mannot: {annot}\ncode: {code}\x1b[0m")
                else: print(f"{i}) {annot}\n{code}")
        
        
if __name__ == "__main__":
    args = get_args()
    demo_annotated_code_search(args)