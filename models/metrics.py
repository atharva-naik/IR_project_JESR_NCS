#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# triplet accuracy model.
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.losses import cos_dist
from collections import defaultdict
        
class RuleWiseAccuracy:
    """rule category wise accuracy."""
    def __init__(self, use_scl: bool=False, margin: int=1):
        self.reset()
        self.margin = margin
        self.use_scl = use_scl
        # if self.use_scl: self.margin = 0
    def reset(self):
        self.counts = defaultdict(lambda:0)
        self.matches = defaultdict(lambda:0)
        
    def update(self, anchor, pos, neg, rule_ids):
        # if self.use_scl:
            # pos = -torch.diag(anchor @ pos.T).cpu() # cos_dist(anchor, pos).cpu()
            # neg = -torch.diag(anchor @ neg.T).cpu() # cos_dist(anchor, neg).cpu()            
        # else:
        pos = F.pairwise_distance(anchor, pos).cpu()
        neg = F.pairwise_distance(anchor, neg).cpu()
        matches = (neg-pos>self.margin)
        for ind, r in enumerate(rule_ids): # print(matches[ind])
            self.counts[r] += 1 
            self.matches[r] += matches[ind].item()
            
    def __str__(self):
        acc = self()
        return "|".join([f"R{r}:{100*acc[i]:.0f}" for i, r in enumerate(sorted(self.counts))])
            
    def __call__(self):
        acc = np.zeros(len(self.counts))
        for i, r in enumerate(self.counts):
            if self.counts[r] == 0: acc[i] = 0
            else: acc[i] = self.matches[r]/self.counts[r]
        return acc

class TripletAccuracy:
    """Triplet accuracy: 1 if the anchor is more similar to the 
    +ve than -ve (successfully separates triplet), 0 otherwise"""
    def __init__(self, use_scl: bool=False, margin: int=1):
        self.margin = margin
        self.reset()
        self.use_scl = use_scl
        self.last_batch_acc = None
        # if self.use_scl: self.margin = 0
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
        # if self.use_scl:
            # pos = -torch.diag(anchor @ pos.T).cpu() # cos_dist(anchor, pos).cpu()
            # neg = -torch.diag(anchor @ neg.T).cpu() # cos_dist(anchor, neg).cpu()
        # else:
        pos = F.pairwise_distance(anchor, pos).cpu()
        neg = F.pairwise_distance(anchor, neg).cpu()
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