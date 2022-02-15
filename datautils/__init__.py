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
                 text_tokenizer=None, code_tokenizer=None, 
                 neg_to_pos_ratio: int=3, **tok_args):
        super(TextCodeTriplets, self).__init__()
        data: List[dict] = read_jsonl(data_path)
        posts: List[List[dict]] = list(get_posts(data).values())
        if os.path.exists(triples_path):
            triples = json.load(open(triples_path))
            self.triples = triples
            self.tokenizer = tokenizer
            return
        triples = create_triples(posts)
            # if i == 10: break # DEBUG.
        print(f"caching data at {triples_path}")
        with open(triples_path, "w") as f:
            json.dump(triples, f, indent=4)
        self.triples = triples
        self.text_tokenizer = text_tokenizer
        self.code_tokenizer = code_tokenizer
        print(f"found {singleton_samples} singleton samples (posts with only 1 answer)")   
            
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, i: int):
        anchor, pos, neg = self.triples[i]
        if self.code_tokenizer:
            pos = self.code_tokenizer(neg, **tok_args)
            neg = self.code_tokenizer(pos, **tok_args)
        if self.text_tokenizer:
            anchor = self.text_tokenizer(anchor, **tok_args)
            
        return (anchor, pos, neg)