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
        self.last_batch_acc = None
        
    def reset(self):
        self.count = 0
        self.tot = 0
        
    def get(self):
        if self.tot == 0: return 0
        else: return self.count/self.tot
        
    def update(self, anchor, pos, neg, mask=None):
        """mask can be a boolean or integer tensor."""
        batch_tot = 0
        batch_count = 0
        pos = self.pdist(anchor, pos).cpu()
        neg = self.pdist(anchor, neg).cpu()
        # print(pos, neg)
        # print("shapes:", pos.shape, neg.shape)
        if mask is not None:
            batch_count += (mask*torch.as_tensor((neg-pos)>self.margin)).sum().item()
            batch_tot += mask.sum().item()
        else:
            batch_count += torch.as_tensor((neg-pos)>self.margin).sum().item()
            batch_tot += len(pos)
        self.count += batch_count
        self.tot += batch_tot
        if batch_tot != 0:
            self.last_batch_acc = batch_count/batch_tot
        else: self.last_batch_acc = 0

# test metrics.
def recall_at_k(actual, predicted, k: int=10):
    rel = 0
    tot = 0
    for act_list, pred_list in zip(actual, predicted):
        for i in act_list:
            tot += 1
            if i in pred_list[:k]: rel += 1
                
    return rel/tot