# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

# -*- coding:utf-8 -*-
import bleu
import json
import argparse
import numpy as np
import syntax_match
from typing import *
import dataflow_match
from tqdm import tqdm
import weighted_ngram_match

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True, 
                    help="a single file with hypothesis & references")
parser.add_argument('--lang', type=str, required=True, 
                    choices=['java','js','c_sharp','php','go','python','ruby'],
                    help='programming language')
parser.add_argument('--params', type=str, default='0.25,0.25,0.25,0.25',
                    help='alpha, beta and gamma')
# file should be JSON
# format: list
# each element of list: Tuple with 2 elements
# 1. 5 candidates
# 2. 1 or more references
# # preprocess inputs
# pre_references = [[x.strip() for x in open(file, 'r', encoding='utf-8').readlines()] \
#                 for file in args.refs]
# hypothesis = [x.strip() for x in open(args.hyp, 'r', encoding='utf-8').readlines()]

# for i in range(len(pre_references)):
#     assert len(hypothesis) == len(pre_references[i])

# references = []
# for i in range(len(hypothesis)):
#     ref_for_instance = []
#     for j in range(len(pre_references)):
#         ref_for_instance.append(pre_references[j][i])
#     references.append(ref_for_instance)
# assert len(references) == len(pre_references)*len(hypothesis)
# calculate weighted ngram match
def make_weights(reference_tokens, key_word_list):
    return {token:1 if token in key_word_list else 0.2 for token in reference_tokens}

def instance_code_bleu(hyp, refs, keywords, tokenizer, 
                       lang="python", params='0.25,0.25,0.25,0.25'):
    # calculate ngram match (BLEU)
    alpha, beta, gamma, theta = [float(x) for x in params.split(',')]
    tokenized_hyps = [tokenizer.tokenize(hyp)]
    tokenized_refs = [[tokenizer.tokenize(x) for x in refs]]
    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)
    tokenized_refs_with_weights = [
        [
            [
                reference_tokens, make_weights(reference_tokens, keywords)
        ] for reference_tokens in reference] for reference in tokenized_refs
    ]
    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights,tokenized_hyps)
    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(refs, [hyp], lang)
    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(refs, [hyp], lang)
    # print('ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}'.\
                        # format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))

    code_bleu_score = alpha*ngram_match_score\
                    + beta*weighted_ngram_match_score\
                    + gamma*syntax_match_score\
                    + theta*dataflow_match_score
    
    return code_bleu_score

def corpus_code_bleu(hypothesis, references, keywords, tokenizer, 
                     lang="python", params='0.25,0.25,0.25,0.25') -> Tuple[float, Dict[str, float]]:
    """calculate corpus level CodeBLEU score."""
    # calculate ngram match (BLEU)
    alpha, beta, gamma, theta = [float(x) for x in params.split(',')]
    tokenized_hyps = [tokenizer.tokenize(x) for x in hypothesis]
    tokenized_refs = [[tokenizer.tokenize(x) for x in reference] for reference in references]
    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)

    # calculate weighted ngram match
    # keywords = [x.strip() for x in open('keywords/'+args.lang+'.txt', 'r', encoding='utf-8').readlines()]
    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)]\
                for reference_tokens in reference] for reference in tokenized_refs]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights,tokenized_hyps)
    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, lang)
    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, lang)
    print('ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}'.\
                        format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))
    code_bleu_score = alpha*ngram_match_score\
                    + beta*weighted_ngram_match_score\
                    + gamma*syntax_match_score\
                    + theta*dataflow_match_score
    
    return code_bleu_score, {
        "ngram_match_score": ngram_match_score,
        "weighted_ngram_match_score": weighted_ngram_match_score,
        "syntax_match_score": syntax_match_score,
        "dataflow_match_score": dataflow_match_score,
    }

if __name__ == "__main__":
    # inst_code_bleu_scores = []
    args = parser.parse_args()
    keywords = [x.strip() for x in open('keywords/'+args.lang+'.txt', 'r', encoding='utf-8').readlines()]
    assert args.file.endswith(".json"), "file should be JSON formatted"
    data = json.load(open(args.file))
    lang = args.lang
    corpus_hyps = []
    corpus_refs = []
    for j, rec in tqdm(enumerate(data), total=len(data)):
        refs = rec[1]
        ret_codes = rec[0]
        ret_code_bleu_scores = []
        for code in ret_codes:
            code_bleu_score = instance_code_bleu(
                code, refs, keywords, lang,
                params=args.params,
            )
            ret_code_bleu_scores.append(code_bleu_score)
        # inst_code_bleu_score 
        i = np.argmax(ret_code_bleu_scores)
        # print(inst_code_bleu_score)
        # inst_code_bleu_scores.append(inst_code_bleu_score)
        corpus_refs.append(refs)
        corpus_hyps.append(ret_codes[i])
    print('CodeBLEU corpus level: ', corpus_code_bleu(corpus_hyps, corpus_refs, keywords, lang, params=args.params))