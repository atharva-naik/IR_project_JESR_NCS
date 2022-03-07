#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import random
from tqdm import tqdm
from typing import List, Dict, Tuple


def read_jsonl(path: str) -> List[dict]:
    """
    read data into list of dicts from a .jsonl file
    """
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            data.append(json.loads(line))
            
    return data

def get_posts(data: List[dict]) -> Dict[str, dict]:
    posts = {} # group the posts by the intent (or post title.)
    # the value contains a list of dataset entries featuring relevant code snippets in the decreasing order of relevance.
    for rec in data:
        intent = rec["intent"]
        try: posts[intent].append(rec)
        except KeyError: posts[intent] = [rec]
            
    return posts

def get_list_complement(l: list, i: int):
    nl = []
    for j in range(len(l)):
        if i == j: continue
        nl.append(l[j])
        
    return nl

def sample_list(l: list, k: int=1):
    """get sampled list."""
    sampled_list = []
    for i in random.sample(range(len(l)), k=k):
        sampled_list.append(l[i])
    
    return sampled_list

def flatten_list(l: List[list]) -> list:
    flat_list = []
    for sub_list in l:
        flat_list += sub_list
        
    return flat_list

def create_triples(posts: List[List[dict]], neg_to_pos_ratio: int=3) -> List[Tuple[str, str, str]]:
    """
    create triples from the list of lists post structure.
    Each post is a list of code snippet answers accompanying the intent (post title).
    The code snippets are actually excerpts of the code blocks that were the answers.
    """
    triples: List[Tuple[str, str, str]] = []
    for i, post in tqdm(enumerate(posts), 
                        desc="creating triplets", 
                        total=len(posts)):
        neg_posts = get_list_complement(posts, i)
        # get list of negative samples from ans of the remaining posts.
        neg_samples = flatten_list(neg_posts)
        singleton_samples = 0
        for j, ans in enumerate(post):
            anchor = ans["intent"]
            # get positive sample from remaining ans of the post.
            pos_posts = get_list_complement(post, j)
            try:
                pos_sample = sample_list(pos_posts, k=1)[0]["snippet"]
            except ValueError:
                singleton_samples += 1
                continue
            # number of negative samples for a given positive sample.
            for neg_post in sample_list(neg_samples, k=neg_to_pos_ratio):
                neg_sample = neg_post["snippet"]
                triples.append((anchor, pos_sample, neg_sample))
                
    return triples

def create_relevant_triples(posts: List[List[dict]], 
                            neg_to_pos_ratio: int=3, 
                            pos_rel_rank_thresh: float=0.25) -> List[Tuple[str, str, str]]:
    """
    create relevant triples from the list of lists post structure.
    Relevant triples are triples made by thresholding by the relevance score,
    while pairing positive code snippet to a given anchor text.
    Each post is a list of code snippet answers accompanying the intent (post title).
    The code snippets are actually excerpts of the code blocks that were the answers.
    """
    triples: List[Tuple[str, str, str]] = []
    for i, post in tqdm(enumerate(posts), 
                        desc="creating triplets",
                        total=len(posts)):
        neg_posts = get_list_complement(posts, i)
        # get list of negative samples from ans of the remaining posts.
        neg_samples = flatten_list(neg_posts)
        singleton_samples = 0
        for j, ans in enumerate(post):
            anchor = ans["intent"]
            # get positive sample from remaining ans of the post.
            pos_posts = get_list_complement(post, j)
            N = max(int(pos_rel_rank_thresh * len(pos_posts)), 1)
            # print(len(pos_posts), N)
            pos_post_rel_tuples = [(rec["prob"], rec) for rec in pos_posts]
            pos_post_rel_tuples = sorted(pos_post_rel_tuples, 
                                         key=lambda x: x[0], 
                                         reverse=True)
            pos_posts = [rec for _,rec in pos_post_rel_tuples[:N]]
            # print(len(pos_posts))
            try:
                pos_sample = sample_list(pos_posts, k=1)[0]["snippet"]
            except ValueError:
                singleton_samples += 1
                continue
            # number of negative samples for a given positive sample.
            for neg_post in sample_list(neg_samples, k=neg_to_pos_ratio):
                neg_sample = neg_post["snippet"]
                triples.append((anchor, pos_sample, neg_sample))
                
    return triples

# TOT = 0
# TOT_NEG_SAMPLES = 0
def get_intra_categ_neg(posts, init: float, 
                        thresh: float, limit: int):
    # global TOT_NEG_SAMPLES
    # global TOT
    egs = []
    num_neg_samples = 0
    for post in posts[::-1]:
        # print(post["prob"], post["prob"]-init, 
        #       thresh, post["prob"] < (init-thresh))
        if (post["prob"] < (init-thresh)):
            egs.append(post)
    # print(f"limit: {limit}", 
    #       TOT_NEG_SAMPLES/TOT, 
    #       f"{len(egs)}/{len(posts)}")
    egs = egs[:limit]
    # TOT_NEG_SAMPLES += len(egs)
    # TOT += len(posts)
    return egs[:limit]
    
def create_triples_intra_categ_neg(posts: List[List[dict]], 
                                   neg_to_pos_ratio: int=3,
                                   intra_categ_thresh: float=0.2) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    for i, post in tqdm(enumerate(posts), 
                        desc="creating triplets", 
                        total=len(posts)):
        neg_posts = get_list_complement(posts, i)
        # get list of negative samples from ans of the remaining posts.
        neg_samples = flatten_list(neg_posts)
        singleton_samples = 0
        for j, ans in enumerate(post):
            anchor = ans["intent"]
            # get positive sample from remaining ans of the post.
            pos_posts = get_list_complement(post, j)
            try:
                sampled_pos_post = sample_list(pos_posts, k=1)[0]
                pos_sample = sampled_pos_post["snippet"]
            except ValueError:
                singleton_samples += 1
                continue
            inter_categ_samples = sample_list(neg_samples, k=neg_to_pos_ratio)
            intra_categ_samples = get_intra_categ_neg(
                pos_posts, thresh=intra_categ_thresh, 
                init=sampled_pos_post["prob"],
                limit=int(0.5*len(post)),
            )
            # number of negative samples for a given positive sample.
            for neg_post in inter_categ_samples+intra_categ_samples:
                neg_sample = neg_post["snippet"]
                triples.append((anchor, pos_sample, neg_sample))
                
    return triples

def create_relevant_triples_intra_categ_neg(posts: List[List[dict]], 
                                            neg_to_pos_ratio: int=3,
                                            intra_categ_thresh: float=0.2,
                                            pos_rel_rank_thresh: float=0.25) -> List[Tuple[str, str, str]]:
    """
    create relevant triples from the list of lists post structure.
    Relevant triples are triples made by thresholding by the relevance score,
    while pairing positive code snippet to a given anchor text.
    Each post is a list of code snippet answers accompanying the intent (post title).
    The code snippets are actually excerpts of the code blocks that were the answers.
    """
    triples: List[Tuple[str, str, str]] = []
    for i, post in tqdm(enumerate(posts), 
                        desc="creating triplets", 
                        total=len(posts)):
        neg_posts = get_list_complement(posts, i)
        # get list of negative samples from ans of the remaining posts.
        neg_samples = flatten_list(neg_posts)
        singleton_samples = 0
        for j, ans in enumerate(post):
            anchor = ans["intent"]
            # get positive sample from remaining ans of the post.
            pos_posts = get_list_complement(post, j)
            # get positive sample from remaining ans of the post.
            pos_posts = get_list_complement(post, j)
            N = max(int(pos_rel_rank_thresh * len(pos_posts)), 1)
            # print(len(pos_posts), N)
            pos_post_rel_tuples = [(rec["prob"], rec) for rec in pos_posts]
            pos_post_rel_tuples = sorted(pos_post_rel_tuples, 
                                         key=lambda x: x[0], 
                                         reverse=True)
            pos_posts = [rec for _,rec in pos_post_rel_tuples[:N]]
            try:
                sampled_pos_post = sample_list(pos_posts, k=1)[0]
                pos_sample = sampled_pos_post["snippet"]
            except ValueError:
                singleton_samples += 1
                continue
            # print(len(pos_posts))
            inter_categ_samples = sample_list(neg_samples, k=neg_to_pos_ratio)
            intra_categ_samples = get_intra_categ_neg(
                pos_posts, thresh=intra_categ_thresh, 
                init=sampled_pos_post["prob"],
                limit=int(0.5*len(post)),
            )
            # number of negative samples for a given positive sample.
            for neg_post in inter_categ_samples+intra_categ_samples:
                neg_sample = neg_post["snippet"]
                triples.append((anchor, pos_sample, neg_sample))
                
    return triples