#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# triplet accuracy model.
import torch
import torch.nn as nn

class TripletAccuracy:
    def __init__(self, margin: int=1):
        self.pdist = nn.PairwiseDistance()
        self.margin = margin
        self.reset()
        
    def reset(self):
        self.count = 0
        self.tot = 0
        
    def get(self):
        if self.tot == 0: return 0
        else: return self.count/self.tot
        
    def update(self, anchor, pos, neg):
        pos = self.pdist(anchor, pos)
        neg = self.pdist(anchor, neg)
        self.count += torch.as_tensor((neg-pos)>self.margin).sum().item()
        self.tot += len(pos)
# test metrics.
def recall_at_k(actual, predicted, k: int=10):
    rel = 0
    tot = 0
    for act_list, pred_list in zip(actual, predicted):
        for i in act_list:
            tot += 1
            if i in pred_list[:k]: rel += 1
                
    return rel/tot