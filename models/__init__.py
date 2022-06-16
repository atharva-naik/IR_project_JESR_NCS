# This package contains main model files, for our "Universal Joint/Shared Space Encoder"
# some common utilites.
import os
import argparse
from typing import *

def get_tok_path(model_name: str) -> str:
    assert model_name in ["codebert", "graphcodebert"]
    if model_name == "codebert":
        tok_path = os.path.expanduser("~/codebert-base-tok")
        if not os.path.exists(tok_path):
            tok_path = "microsoft/codebert-base"
    elif model_name == "graphcodebert":
        tok_path = os.path.expanduser("~/graphcodebert-base-tok")
        if not os.path.exists(tok_path):
            tok_path = "microsoft/grapcodebert-base"
            
    return tok_path

def test_ood_performance(model_name: str, query_paths: List[str], 
                         cand_paths: List[str], args: argparse.Namespace):
    # do only code retrieval with l2 distance as distance function
    print("initializing model and tokenizer ..")
    tok_path = get_tok_path(model_name)
    device = args.device_id if torch.cuda.is_available() else "cpu"
    
    ckpt_path = os.path.join(args.exp_name, "model.pt")
    print(f"loading checkpoint (state dict) from {ckpt_path}")
    try: state_dict = torch.load(ckpt_path, map_location="cpu")
    except Exception as e: 
        state_dict = None
        print("Couldn't load state dict because:")
        print(e)
    
    print("creating model object")
    if model_name == "codebert":
        triplet_net = CodeBERTripletNet(tok_path=tok_path, **vars(args))
    elif model_name == "graphcodebert":
        triplet_net = GraphCodeBERTripletNet(tok_path=tok_path, **vars(args))
    if state_dict: 
        print(f"\x1b[32;1mloading state dict from {ckpt_path}\x1b[0m")
        triplet_net.load_state_dict(state_dict)
    for query_path, cand_path in zip(query_paths, cand_paths):
        # load code candidates.
        print(f"loading candidates from {cand_path}")
        code_and_annotations = json.load(open(cand_path))
        candidates = code_and_annotations["snippets"]

        print(f"loading queries from {query_path}")
        queries_and_cand_labels = json.load(open(query_path))
        queries = [i["query"] for i in queries_and_cand_labels]
        labels = [i["docs"] for i in queries_and_cand_labels]
        # distance function to be used.
        dist_fn = args.dist_fn
        assert dist_fn in ["l2_dist", "inner_prod"]
        metrics_path = os.path.join(
            args.exp_name, 
            f"test_metrics_{dist_fn}_{setting}.json"
        )
        # encode queries.
        print(f"encoding {len(queries)} queries:")
        query_mat = triplet_net.encode_emb(queries, mode="text", 
                                           batch_size=args.batch_size,
                                           use_tqdm=True, device_id=device)
        query_mat = torch.stack(query_mat)
        # encode candidates.
        print(f"encoding {len(candidates)} candidates:")
        if setting == "code":
            cand_mat = triplet_net.encode_emb(candidates, mode="code", 
                                              batch_size=args.batch_size,
                                              use_tqdm=True, device_id=device)
            cand_mat = torch.stack(cand_mat)
        elif setting == "annot":
            cand_mat = triplet_net.encode_emb(candidates, mode="text", 
                                              batch_size=args.batch_size,
                                              use_tqdm=True, device_id=device)
            cand_mat = torch.stack(cand_mat)
        else:
            cand_mat_code = triplet_net.encode_emb(code_candidates, mode="code", 
                                                   batch_size=args.batch_size,
                                                   use_tqdm=True, device_id=device)
            cand_mat_annot = triplet_net.encode_emb(annot_candidates, mode="text",
                                                    batch_size=args.batch_size,
                                                    use_tqdm=True, device_id=device)
            cand_mat_code = torch.stack(cand_mat_code)
            cand_mat_annot = torch.stack(cand_mat_annot)
                # cand_mat = (cand_mat_code + cand_mat_annot)/2
        # print(query_mat.shape, cand_mat.shape)
        if dist_func == "inner_prod": 
            if setting == "code+annot":
                scores_code = query_mat @ cand_mat_code.T
                scores_annot = query_mat @ cand_mat_annot.T
                scores = scores_code + scores_annot
            else:
                scores = query_mat @ cand_mat.T
            # print(scores.shape)
        elif dist_func == "l2_dist": 
            if setting == "code+annot":
                scores_code = torch.cdist(query_mat, cand_mat_code, p=2)
                scores_annot = torch.cdist(query_mat, cand_mat_annot, p=2)
                scores = scores_code + scores_annot
            else:
                scores = torch.cdist(query_mat, cand_mat, p=2)
        # elif mode == "joint_cls": scores = triplet_net.joint_classify(queries, candidates)
        doc_ranks = scores.argsort(axis=1)
        if dist_func == "inner_prod":
            doc_ranks = doc_ranks.flip(dims=[1])
        label_ranks = []
        avg_rank = 0
        avg_best_rank = 0 
        N = 0
        M = 0

        lrap_GT = np.zeros(
            (
                len(queries), 
                len(candidates)
            )
        )
        recall_at_ = []
        for i in range(1,10+1):
            recall_at_.append(
                recall_at_k(
                    labels, 
                    doc_ranks.tolist(), 
                    k=5*i
                )
            )
        for i in range(len(labels)):
            for j in labels[i]:
                lrap_GT[i][j] = 1

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
        metrics = {
            "avg_candidate_rank": avg_rank/N,
            "avg_best_candidate_rank": avg_best_rank/M,
            "recall": {
                f"@{5*i}": recall_at_[i-1] for i in range(1,10+1) 
            },
        }
        print("avg canditate rank:", avg_rank/N)
        print("avg best candidate rank:", avg_best_rank/M)
        for i in range(1,10+1):
            print(f"recall@{5*i} = {recall_at_[i-1]}")
        if dist_func == "inner_prod":
            # -scores for distance based scores, no - for innert product based scores.
            mrr = MRR(lrap_GT, scores.cpu().numpy())
            ndcg = NDCG(lrap_GT, scores.cpu().numpy())
        elif dist_func == "l2_dist":
            # -scores for distance based scores, no - for innert product based scores.
            mrr = MRR(lrap_GT, -scores.cpu().numpy())
            ndcg = NDCG(lrap_GT, -scores.cpu().numpy())

        metrics["mrr"] = mrr
        metrics["ndcg"] = ndcg
        print("NDCG:", ndcg)
        print("MRR (LRAP):", mrr)
        if not os.path.exists(args.exp_name):
            print("missing experiment folder: assuming zero-shot setting")
            metrics_path = os.path.join(
                "CodeBERT_zero_shot", 
                f"test_metrics_{dist_func}_{setting}.json"
            )
            os.makedirs("CodeBERT_zero_shot", exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)