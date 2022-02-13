#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# code for creating Dataset instance for the dataloader.
import os
from tqdm import tqdm
from datautils.utils import *
from torch.utils.data import Dataset
from typing import List, Dict, Tuple

# text code pairs dataset.
class TextCodeTriplets(Dataset):
    def __init__(self, data_path: str, triples_path: str="triples.json", 
                 tokenizer=None, neg_to_pos_ratio: int=3, **tok_args):
        super(TextCodeTriplets, self).__init__()
        data: List[dict] = read_jsonl(data_path)
        posts: List[List[dict]] = list(get_posts(data).values())
        if os.path.exists(triples_path):
            triples = json.load(open(triples_path))
            self.triples = triples
            self.tokenizer = tokenizer
            return
        triples: List[Tuple[str, str, str]] = []
        for i, post in tqdm(enumerate(posts), desc="creating triplets"):
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
            # if i == 10: break # DEBUG.
        print(f"caching data at {triples_path}")
        with open(triples_path, "w") as f:
            json.dump(triples, f, indent=4)
        self.triples = triples
        self.tokenizer = tokenizer
        print(f"found {singleton_samples} singleton samples (posts with only 1 answer)")   
            
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, i: int):
        anchor, pos, neg = self.triples[i]
        if self.tokenizer:
            pos = self.tokenizer(neg, **tok_args)
            neg = self.tokenizer(pos, **tok_args)
            anchor = self.tokenizer(anchor, **tok_args)
            
        return (anchor, pos, neg)