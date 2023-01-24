#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
import os
import json
import argparse
import tokenize
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer
from code_bleu import instance_code_bleu, corpus_code_bleu

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--k", type=int, default=5, help="recall@k used to find the error cases")
parser.add_argument("-e1", "--exp1", required=True, help="name of the experiment (1st model)") # the better model.
parser.add_argument("-e2", "--exp2", required=True, help="name of the experiment (2nd model)") # the worse model.
# parser.add_argument("-d", "--dataset", type=str, required=True, help="the dataset")
# should be in the approved list of datasets.
# assert args.dataset in ["CoNaLa", "PyDocs", "WebQuery"]
def get_model_type(exp_name: str) -> str:
    exp_name = exp_name.lower()
    if "unixcoder" in exp_name: return "UX"
    elif "codebert" in exp_name: return "CB"
    elif "graphcodebert" in exp_name: return "GCB"
    
def get_approach_type(exp_name: str) -> str:
    exp_name = exp_name.lower()
    if "_ast_" in exp_name: return "AST"
    elif "_disco" in exp_name: return "DISCO"
    elif "_coderetriever" in exp_name: return "CR"

def get_short_name(exp_name: str) -> str:
    return get_model_type(exp_name)+"+"+get_approach_type(exp_name)
    
def get_data(args, datasets: list):
    preds1 = []
    preds2 = []
    candidates = []
    trues = []
    queries = []
    for dataset in datasets:
        test_path = {
            "CoNaLa": "query_and_candidates.json", 
            "PyDocs": "external_knowledge/queries.json",
            "WebQuery": "data/queries_webquery.json"}[dataset]
        canidates_path = {
            "CoNaLa": "candidate_snippets.json", 
            "PyDocs": "external_knowledge/candidates.json",
            "WebQuery": "data/candidates_webquery.json"}[dataset]
        exp_path_map = {
            "CoNaLa": "CoNaLa Doc Ranks.json",
            "PyDocs": "External Knowledge Doc Ranks.json",
            "WebQuery": "Web Query Doc Ranks.json",
        }
        exp_path1 = os.path.join("experiments", args.exp1, 
                                 exp_path_map[dataset])
        exp_path2 = os.path.join("experiments", args.exp2, 
                                 exp_path_map[dataset])
        path = os.path.join("improvement_egs", f"{args.exp1}_minus_{args.exp2}_{dataset}.json") 
        for rec in json.load(open(exp_path1)):
            preds1.append(rec)
        for rec in json.load(open(exp_path2)):
            preds2.append(rec)
        for rec in json.load(open(canidates_path))['snippets']:
            candidates.append(rec)
        for rec in json.load(open(test_path)):
            trues.append(rec['docs'])
            queries.append(rec['query'])
    
    return preds1, preds2, candidates, trues, queries

def compare_CodeBLEU(args, datasets: list, subset: str, tokenizer, lang: str="python"):
    assert subset in ["ID", "OOD"], "subset not known"
    keywords = [x.strip() for x in open('CodeBLEU/keywords/'+lang+'.txt', 'r', encoding='utf-8').readlines()]
    k = args.k
    preds1, preds2, candidates, trues, queries = get_data(args, datasets)
    bmgc_wmgw = 0
    bmgw_wmgc = 0
    bm_hypotheses = [] # better model hypotheses.
    wm_hypotheses = [] # worst model hypotheses.
    references = [] # model references.
    bm_bmgc_wmgw_hyps = []
    bm_bmgw_wmgc_hyps = []
    bm_missed_hyps = []
    bm_missed_refs = []
    bmgc_wmgw_refs = []
    wm_bmgc_wmgw_hyps = []
    wm_bmgw_wmgc_hyps = []
    wm_missed_hyps = []
    wm_missed_refs = []
    bmgw_wmgc_refs = []
    for i in tqdm(range(len(preds1))):
        sel_hyp1 = None
        sel_hyp2 = None
        pred_row1 = preds1[i]
        pred_row2 = preds2[i]
        hyps1 = [candidates[ind] for ind in pred_row1]
        hyps2 = [candidates[ind] for ind in pred_row2]
        labels = trues[i]
        query = queries[i]
        bm_all_scores = [] # instance CodeBLEU scores for better model (1)
        wm_all_scores = [] # instance CodeBLEU scores for worse model (2)
        worse_model_gets_correct = False
        better_model_gets_correct = False
        # check if better model gets correct, or worse model gets correct.
        for label in labels:
            if label in pred_row1[:k]:
                better_model_gets_correct = True
                sel_hyp1 = candidates[label]
                break
        for label in labels:
            if label in pred_row2[:k]:
                worse_model_gets_correct = True
                sel_hyp2 = candidates[label]
                break
        refs = [candidates[ind] for ind in labels]
        # print(f"\x1b[34;1m105\x1b[0m: {refs}")
        # better model.
        if sel_hyp1 is None:
            bm_missed_refs.append(refs)
            for ind in pred_row1[:k]:
                bm_code = hyps1[ind]
                code_bleu_score = instance_code_bleu(bm_code, refs, keywords, tokenizer)
                # print(f"\x1b[34;1m110\x1b[0m: {refs}")
                bm_all_scores.append(code_bleu_score)
            sel_hyp1 = hyps1[np.argmax(bm_all_scores)]
            bm_missed_hyps.append(sel_hyp1)
        # worse model.
        if sel_hyp2 is None:
            wm_missed_refs.append(refs)
            for ind in pred_row2[:k]:
                wm_code = hyps2[ind]
                code_bleu_score = instance_code_bleu(wm_code, refs, keywords, tokenizer)
                # print(f"\x1b[34;1m116\x1b[0m: {refs}")
                wm_all_scores.append(code_bleu_score)
            sel_hyp2 = hyps2[np.argmax(wm_all_scores)]
            wm_missed_hyps.append(sel_hyp2)
        bm_hypotheses.append(sel_hyp1)
        wm_hypotheses.append(sel_hyp2)
        references.append(refs)
        if better_model_gets_correct and not worse_model_gets_correct:
            codes1 = [candidates[l] for l in pred_row1[:k]]
            codes2 = [candidates[l] for l in pred_row2[:k]]
            true_codes = [candidates[l] for l in labels]
            bmgc_wmgw += 1
            bm_bmgc_wmgw_hyps.append(sel_hyp1)
            wm_bmgc_wmgw_hyps.append(sel_hyp2)
            bmgc_wmgw_refs.append(refs)
        if not better_model_gets_correct and worse_model_gets_correct:
            bmgw_wmgc += 1
            bm_bmgw_wmgc_hyps.append(sel_hyp1)
            wm_bmgw_wmgc_hyps.append(sel_hyp2)
            bmgw_wmgc_refs.append(refs)

    bm_bmgw_wmgc_score, bm_bmgw_wmgc_breakdown  = corpus_code_bleu(bm_bmgw_wmgc_hyps, bmgw_wmgc_refs, keywords, tokenizer)
    bm_bmgc_wmgw_score, bm_bmgc_wmgw_breakdown = corpus_code_bleu(bm_bmgc_wmgw_hyps, bmgc_wmgw_refs, keywords, tokenizer)
    wm_bmgw_wmgc_score, wm_bmgw_wmgc_breakdown = corpus_code_bleu(wm_bmgw_wmgc_hyps, bmgw_wmgc_refs, keywords, tokenizer)
    wm_bmgc_wmgw_score, wm_bmgc_wmgw_breakdown = corpus_code_bleu(wm_bmgc_wmgw_hyps, bmgc_wmgw_refs, keywords, tokenizer)
    bm_score, bm_breakdown = corpus_code_bleu(bm_hypotheses, references, keywords, tokenizer)
    wm_score, wm_breakdown = corpus_code_bleu(wm_hypotheses, references, keywords, tokenizer)
    bm_missed_score, bm_missed_breakdown = corpus_code_bleu(bm_missed_hyps, bm_missed_refs, keywords, tokenizer)
    wm_missed_score, wm_missed_breakdown = corpus_code_bleu(wm_missed_hyps, wm_missed_refs, keywords, tokenizer)
    n1 = get_short_name(args.exp1)
    n2 = get_short_name(args.exp2)
    print(f"{n1} CodeBLEU: {bm_score}, {n2} CodeBLEU: {wm_score}")
    print(f"{n1} gets correct but {n2} gets wrong: {bmgc_wmgw}, {n1} CodeBELU: {bm_bmgc_wmgw_score}, {n2} CodeBELU: {wm_bmgc_wmgw_score}")
    print(f"{n1} gets wrong but {n2} gets correct: {bmgw_wmgc}, {n1} CodeBLEU: {bm_bmgw_wmgc_score}, {n2} CodeBLEU: {wm_bmgw_wmgc_score}")
    all_scores = {
        n1: {
            "total": (bm_score, bm_breakdown),
            "missed_preds": (bm_missed_score, bm_missed_refs), # missed in top 5 hits.
            "does_better": (bm_bmgc_wmgw_score, bm_bmgc_wmgw_breakdown),
            "does_worse": (bm_bmgw_wmgc_score, bm_bmgw_wmgc_breakdown),
        },
        n2: {
            "total": (wm_score, wm_breakdown),
            "missed_preds": (wm_missed_score, wm_missed_refs), # missed in top 5 hits.
            "does_better": (wm_bmgc_wmgw_score, wm_bmgc_wmgw_breakdown),
            "does_worse": (wm_bmgw_wmgc_score, wm_bmgw_wmgc_breakdown),
        }
    }
    save_path = f"{n1}_{n2}_code_bleu_{subset}_comparison.json"
    with open(save_path, "w") as f:
        json.dump(all_scores, f, indent=4)
    
class CodeTokenizer:
    def tokenize(self, text):
        text = text.strip()
        io_obj = io.StringIO(text)
        tokens = []
        try:
            token_gen_exp = tokenize.generate_tokens(io_obj.readline)
            for tok in token_gen_exp:
                tok = tok.string.strip()
                if tok != "": tokens.append(tok)
            return tokens
        except tokenize.TokenError: return text.split()
    
if __name__ == '__main__':
    args = parser.parse_args()
    tok_path = os.path.expanduser("~/unixcoder-base-tok")
    # tokenizer = RobertaTokenizer.from_pretrained(tok_path)
    tokenizer = CodeTokenizer()
    print("ID:")
    compare_CodeBLEU(args, ["CoNaLa"], "ID", tokenizer)
    print("OOD:")
    compare_CodeBLEU(args, ["PyDocs", 'WebQuery'], "OOD", tokenizer)
# scripts/find_improvement_egs.py -e1 GraphCodeBERT_ast_5_100k -e2 GraphCodeBERT_100k -m GraphCodeBERT -d PyDocs -k 5
# scripts/find_improvement_egs.py -e1 GraphCodeBERT_ast_5_100k -e2 GraphCodeBERT_100k -m GraphCodeBERT -d WebQuery -k 5