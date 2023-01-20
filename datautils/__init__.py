#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Atharva Naik (18CS10067)

# code for creating Dataset instance for the dataloader.
import os
import copy
import torch
import random
import numpy as np
from typing import *
from tqdm import tqdm
import torch.nn as nn
from datautils.utils import *
import torch.nn.functional as F
from collections import defaultdict
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
from scripts.create_code_code_pairs import CodeSynsets

# list of available models. 
WORST_RULES_LIST = ["rule1", "rule3", "rule8", "rule11", "rule13", "rule17"] # the worst 6 rules
WORST_OLD_RULES_LIST = ["rule1", "rule2", "rule3", "rule8"] # the worst 4 (out of 9 rules) according to the old system.
DISCO_IGNORE_LIST = ["rule1", "rule2", "rule3", "rule5", "rule7", 
                     "rule8", "rule16", "rule17", "rule18"] # rules to be ignore to follow DISCO
UNNATURAL_IGNORE_LIST = ["rule8", "rule12", "rule13", "rule14", "rule15", "rule17"]
NEW_RULES_IGNORE_LIST = ["rule10", "rule11", "rule14", "rule15", "rule16", "rule17", "rule18"]
MODEL_OPTIONS = ["codebert", "graphcodebert", "unixcoder"]
#     anchor["input_ids"][0], anchor["attention_mask"][0], 
#     pos["input_ids"][0], pos["attention_mask"][0],
#     neg["input_ids"][0], neg["attention_mask"][0],
#     torch.tensor(hard_neg),
# ]
def batch_shuffle_collate_fn_graphcodebert(batch):
    is_hard_neg_mask = []
    # a mapping to shuffle indices of soft negatives
    # i.e. assign positive code snippets from one triplet as negative code snippets for other triplets
    # but only for soft negatives. Leave the hard negatives as it is.
    shuffle_map = {}
    # positive PL related inputs:
    pos_0 = [] # positive PL batch indices.
    pos_1 = []
    pos_2 = []
    # negative PL related inputs.
    neg_3 = [] # negative PL batch indices.
    neg_4 = []
    neg_5 = []
    anchor = [] # the 6th index in the batch.
    is_hard_neg_mask = []
    for i in range(len(batch)):
        # whether current triplet is a hard negative.
        is_hard_neg = batch[i][-1].item()
        # populate the mapping of soft indices as an identity map first.
        if not is_hard_neg:
            shuffle_map[i] = i
        is_hard_neg_mask.append(is_hard_neg)
        anchor.append(batch[i][6])
        pos_0.append(batch[i][0])
        pos_1.append(batch[i][1])
        pos_2.append(batch[i][2])
    # determine new indices by random shuffling.
    new_inds = random.sample(list(shuffle_map.values()), k=len(shuffle_map))
    for old_ind, new_ind in zip(shuffle_map, new_inds):
        shuffle_map[old_ind] = new_ind
    # create the negative samples while modifying soft negatives using the shuffle map
    # but leave the hard negative triplets unaffected.
    for i in range(len(batch)):
        is_hard_neg = batch[i][-1].item()
        if is_hard_neg: # for hard negatives don't modify triplet
            neg_3.append(batch[i][3])
            neg_4.append(batch[i][4])
            neg_5.append(batch[i][5])
        else: # for soft negative triplets use the shuffle map to get the positive code snippet from a different triplet.
            use_ind = shuffle_map[i]
            neg_3.append(batch[use_ind][0]) 
            neg_4.append(batch[use_ind][1])
            neg_5.append(batch[use_ind][2])
    
    pos_0 = torch.stack(pos_0)
    pos_1 = torch.stack(pos_1)
    pos_2 = torch.stack(pos_2)
    neg_3 = torch.stack(neg_3)
    neg_4 = torch.stack(neg_4)
    neg_5 = torch.stack(neg_5)
    anchor = torch.stack(anchor)
    is_hard_neg_mask = torch.as_tensor(is_hard_neg_mask)
    
    return [pos_0, pos_1, pos_2, neg_3, neg_4, neg_5, anchor, is_hard_neg_mask]

def batch_shuffle_collate_fn_codebert(batch):
    is_hard_neg_mask = []
    # a mapping to shuffle indices of soft negatives
    # i.e. assign positive code snippets from one triplet as negative code snippets for other triplets
    # but only for soft negatives. Leave the hard negatives as it is.
    shuffle_map = {}
    anchor = [] # the anchor NL batch indices.
    anchor_attn = []
    pos = [] # positive PL batch indices.
    pos_attn = []
    neg = [] # negative PL batch indices.
    neg_attn = []
    is_hard_neg_mask = []
    for i in range(len(batch)):
        # whether current triplet is a hard negative.
        is_hard_neg = batch[i][-1].item()
        # populate the mapping of soft indices as an identity map first.
        if not is_hard_neg:
            shuffle_map[i] = i
        is_hard_neg_mask.append(is_hard_neg)
        anchor.append(batch[i][0])
        anchor_attn.append(batch[i][1])
        pos.append(batch[i][2])
        pos_attn.append(batch[i][3])
    # determine new indices by random shuffling.
    new_inds = random.sample(list(shuffle_map.values()), k=len(shuffle_map))
    for old_ind, new_ind in zip(shuffle_map, new_inds):
        shuffle_map[old_ind] = new_ind
    # create the negative samples while modifying soft negatives using the shuffle map
    # but leave the hard negative triplets unaffected.
    for i in range(len(batch)):
        is_hard_neg = batch[i][-1].item()
        if is_hard_neg: # for hard negatives don't modify triplet
            neg.append(batch[i][4])
            neg_attn.append(batch[i][5])
        else: # for soft negative triplets use the shuffle map to get the positive code snippet from a different triplet.
            use_ind = shuffle_map[i]
            neg.append(batch[use_ind][2]) # '2' as we are using positive code snippet from another triplet.
            neg_attn.append(batch[use_ind][3])
    anchor = torch.stack(anchor)
    pos = torch.stack(pos)
    neg = torch.stack(neg)
    anchor_attn = torch.stack(anchor_attn)
    pos_attn = torch.stack(pos_attn)
    neg_attn = torch.stack(neg_attn)
    is_hard_neg_mask = torch.as_tensor(is_hard_neg_mask)
    
    return [anchor, anchor_attn, pos, pos_attn, neg, neg_attn, is_hard_neg_mask]

def batch_shuffle_collate_fn(batch):
    is_hard_neg_mask = []
    # a mapping to shuffle indices of soft negatives
    # i.e. assign positive code snippets from one triplet as negative code snippets for other triplets
    # but only for soft negatives. Leave the hard negatives as it is.
    shuffle_map = {}
    anchor = [] # the anchor NL batch indices.
    pos = [] # positive PL batch indices.
    neg = [] # negative PL batch indices.
    is_hard_neg_mask = []
    for i in range(len(batch)):
        # whether current triplet is a hard negative.
        is_hard_neg = batch[i][3].item()
        # populate the mapping of soft indices as an identity map first.
        if not is_hard_neg:
            shuffle_map[i] = i
        is_hard_neg_mask.append(is_hard_neg)
        anchor.append(batch[i][0])
        pos.append(batch[i][1])
    # determine new indices by random shuffling.
    new_inds = random.sample(list(shuffle_map.values()), k=len(shuffle_map))
    for old_ind, new_ind in zip(shuffle_map, new_inds):
        shuffle_map[old_ind] = new_ind
    # create the negative samples while modifying soft negatives using the shuffle map
    # but leave the hard negative triplets unaffected.
    for i in range(len(batch)):
        is_hard_neg = batch[i][3].item()
        if is_hard_neg: # for hard negatives don't modify triplet
            neg.append(batch[i][2])
        else: # for soft negative triplets use the shuffle map to get the positive code snippet from a different triplet.
            use_ind = shuffle_map[i]
            neg.append(batch[use_ind][1]) # '1' as we are using positive code snippet from another triplet.
    anchor = torch.stack(anchor)
    pos = torch.stack(pos)
    neg = torch.stack(neg)
    is_hard_neg_mask = torch.as_tensor(is_hard_neg_mask)
    
    return [anchor, pos, neg, is_hard_neg_mask]
    
# dynamic dataloader class: has custom collating function that can use IDNS if needed.
class DynamicDataLoader(DataLoader):
    def __init__(self, *args, device: str="cpu", 
                 model=None, model_name="unixcoder", **kwargs):
        self.device = device
        # model and model device.
        self.model = model
        super(DynamicDataLoader, self).__init__(
            *args, collate_fn=self.collate_fn, **kwargs,
        )
        
    def collate_fn(self, batch: List[tuple]):
        is_hard_neg_mask = []
        # a mapping to shuffle indices of soft negatives
        # i.e. assign positive code snippets from one triplet as negative code snippets for other triplets
        # but only for soft negatives. Leave the hard negatives as it is.
        shuffle_map = {}
        soft_neg_queries = []
        soft_neg_cands = []
        anchor = [] # the anchor NL batch indices.
        pos = [] # positive PL batch indices.
        neg = [] # negative PL batch indices.
        is_hard_neg_mask = []
        for i in range(len(batch)):
            # whether current triplet is a hard negative.
            is_hard_neg = batch[i][3].item()
            # populate the mapping of soft indices as an identity map first.
            if not is_hard_neg:
                shuffle_map[i] = i
                soft_neg_cands.append(batch[i][1])
                soft_neg_queries.append(batch[i][0])
            is_hard_neg_mask.append(is_hard_neg)
            anchor.append(batch[i][0])
            pos.append(batch[i][1])
        # re-order the soft negatives stored in the shuffle map.
        self.model.eval()
        with torch.no_grad():
            soft_neg_cands = torch.stack(soft_neg_cands)
            soft_neg_queries = torch.stack(soft_neg_queries)
            _, enc_intents = self.model(soft_neg_queries.to(self.device))
            _, enc_snippets = self.model(soft_neg_cands.to(self.device))
            # score negatives and get there ranks. No mask required because self map is also valid.
            scores = enc_intents @ enc_snippets.T # batch_size x batch_size
            ranks = torch.topk(scores, k=1, axis=1).indices.T.squeeze() # k x batch_size
            # print(ranks.shape)
        # determine new indices by random shuffling.
        new_shuffle_map = {}
        keys = list(shuffle_map.keys())
        for old_ind, reorder_ind in zip(shuffle_map, ranks):
            new_shuffle_map[old_ind] = shuffle_map[keys[reorder_ind.item()]]
        shuffle_map = new_shuffle_map
        # create the negative samples while modifying soft negatives using the shuffle map
        # but leave the hard negative triplets unaffected.
        for i in range(len(batch)):
            is_hard_neg = batch[i][3].item()
            if is_hard_neg: # for hard negatives don't modify triplet
                neg.append(batch[i][2])
            else: # for soft negative triplets use the shuffle map to get the positive code snippet from a different triplet.
                use_ind = shuffle_map[i]
                neg.append(batch[use_ind][1]) # '1' as we are using positive code snippet from another triplet.
        # self.model.train()
        anchor = torch.stack(anchor)
        pos = torch.stack(pos)
        neg = torch.stack(neg)
        is_hard_neg_mask = torch.as_tensor(is_hard_neg_mask)

        return [anchor, pos, neg, is_hard_neg_mask]
    
# 1-D float buffer and mastering rate calculation.
class MasteringRate:
    def __init__(self, role: int=0, size: int=20, 
                 p: float=2, delta: float=0.5):
        self.acc_buffer = np.zeros(size)
        self.window_size = size
        self.delta = delta
        self.p = p
        self.role = role # role of 0 corresponds to soft accuracy and hard corresponds to 1
        self.window_mean = 0
        self.mean_size = (size+1)/2
        
    def __call__(self):
        return self.window_mean
        
    def __repr__(self):
        return repr(self.acc_buffer)
    
    def __str__(self):
        return f"{self.role}:[{self.window_size}]:={self.window_mean:.3f}(p={self.p}, δ={self.delta})"
        
    def beta(self):
        """Calculate slope of linear regression for returns/accuracies
        for the attention computation."""
        t = 1+np.array(range(self.window_size))
        M = self.mean_size + np.zeros(self.window_size)
        NUM = (t-M)*(self.acc_buffer-self.window_mean)
        DENOM = ((t-M)**2)
        
        return NUM.sum()/DENOM.sum()
        
    def attn(self, other):
        """calculate the attention for the task from the mastering rate.
        `other_master_rate`: the window mean of the other master rate object."""
        beta = self.beta()
        beta_max = max(beta, other.beta())
        if beta_max != 0: beta = beta/beta_max
        if self.role == 0:
            return (self.delta*(1-self.window_mean) + (1-self.delta)*beta)*(1-other.window_mean)
        else:
            return (other.window_mean**self.p)*(self.delta*(1-self.window_mean) + (1-self.delta)*beta)
        
    def push_acc(self, acc: float):
        for i in range(self.window_size-1):
            self.acc_buffer[i] = self.acc_buffer[i+1]
        self.acc_buffer[-1] = acc
        self.window_mean = self.acc_buffer.mean()
# class DynamicTriplesDataset(Dataset):
#     def __init__(self, path: str, model_name: str, model=None, tokenizer=None,
#                  use_AST=False, val=False, warmup_steps=3000, beta=0.001, p=2,
#                  sim_intents_map={}, perturbed_codes={}, device="cuda:0", win_size=20, 
#                  delta=0.5, epsilon=0.8, use_curriculum=True, curriculum_type="mr", 
#                  soft_neg_bias=0.8, batch_size=None, num_epochs=None, **tok_args):
#         super(DynamicTriplesDataset, self).__init__()
#         self.use_curriculum = use_curriculum
#         # print(f"\x1b[31;1m{curriculum_type}\x1b[0m")
#         # self.rand_curriculum = rand_curriculum
#         # check if valid curriculum type:
#         msg = f"invalid curriculum type: {curriculum_type}"
#         assert curriculum_type in ["mr", "rand", "lp", 
#                                    "exp", "hard", "soft"], msg
#         self.curriculum_type = curriculum_type
#         if curriculum_type != "mr": 
#             warmup_steps = 0
#         assert model_name in MODEL_OPTIONS
#         self.step_ctr = 0
#         self.num_epochs = num_epochs
#         self.batch_size = batch_size
#         self.model_name = model_name
#         self.warmup_steps = warmup_steps
#         # self.milestone_updater = MilestoneUpdater()
#         self.p = p
#         self.beta = beta
#         self.epsilon = epsilon
#         self.soft_master_rate = MasteringRate(
#             role=0, delta=delta,
#             p=p, size=win_size, 
#         )
#         self.hard_master_rate = MasteringRate(
#             role=1, delta=delta,
#             p=p, size=win_size, 
#         )
#         self.soft_neg_bias = soft_neg_bias
#         self.soft_neg_weight = 0.8
#         self.hard_neg_weight = 0.2
#         if self.curriculum_type == "hard":
#             self.soft_neg_weight = 0
#             self.hard_neg_weight = 1
#         elif self.curriculum_type == "soft":
#             self.soft_neg_weight = 1
#             self.hard_neg_weight = 0
#         self.model = model # pointer to model instance to find closest NL & PL examples
#         self.sim_intents_map = sim_intents_map
#         self.perturbed_codes = perturbed_codes
#         self.tok_args = tok_args
#         self.use_AST = use_AST
#         self.device = device
#         self.val = val
#         self.lp_s = 0
#         self.lp_h = 0
#         # if filename endswith jsonl:
#         if path.endswith(".jsonl"):
#             self.data = read_jsonl(path) # NL-PL pairs.
#         # if filename endswith json:
#         elif path.endswith(".json"):
#             self.data = json.load(open(path)) # NL-PL pairs.
#         # create a mapping of NL to all associated PLs. 
#         self.intent_to_code = {}
#         for rec in self.data:
#             try: intent = rec["intent"]
#             except TypeError: intent = rec[0]
#             try: snippet = rec["snippet"]
#             except TypeError: snippet = rec[1]
#             try: self.intent_to_code[intent].append(snippet)
#             except KeyError: self.intent_to_code[intent] = [snippet]
#         # parser is needed for GraphCodeBERT to get the dataflow.
#         if model_name == "graphcodebert":
#             from datautils.parser import DFG_python
#             from tree_sitter import Language, Parser
#             PARSER =  Parser()
#             LANGUAGE = Language('datautils/py_parser.so', 'python')
#             PARSER.set_language(LANGUAGE)
#             self.parser = [PARSER, DFG_python]
#         if isinstance(tokenizer, RobertaTokenizer): 
#             self.tokenizer = tokenizer
#         elif isinstance(tokenizer, str):
#             self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
#         else: self.tokenizer = tokenizer
#         if curriculum_type == "exp":
#             assert batch_size is not None, "need batch size for exponential decay curriculum"
#             assert num_epochs is not None, "need num epochs for exponential decay curriculum"
#             N = len(self)*self.num_epochs
#             half_life_frac = 0.8 # this means the weights will be halved when 50% of training is complete
#             self.Z = N*half_life_frac
#             print("using exponentialy decaying curriculum")
        
#     def update(self, soft_acc: float, hard_acc: float):
#         # self.milestone_updater.update(acc)
#         self.soft_master_rate.push_acc(soft_acc)
#         self.hard_master_rate.push_acc(hard_acc)
#         if self.curriculum_type == "mr":
#             if self.warmup_steps > 0: self.warmup_steps -= 1; return 
#             a_s = self.soft_master_rate.attn(self.hard_master_rate)
#             a_h = self.hard_master_rate.attn(self.soft_master_rate)
#             attn_dist = F.softmax(torch.as_tensor([a_s, a_h]), dim=0)
#             bias_dist = torch.as_tensor([self.soft_neg_bias, 1-self.soft_neg_bias])
#             weights = (1-self.epsilon)*attn_dist + self.epsilon*bias_dist
#             weights = weights.numpy()
#         elif self.curriculum_type == "exp":
#             self.step_ctr += self.batch_size
#             r = self.step_ctr/self.Z
#             weights = [np.exp(-np.log(2)*r)]
#         elif self.curriculum_type == "lp":
#             lp_s = self.soft_master_rate()
#             lp_h = self.hard_master_rate()
#             a_s = (1-lp_s)*(1-lp_h)
#             a_h = (lp_s**self.p)*(1-lp_h)
#             self.lp_s, self.lp_h = lp_s, lp_h
#             weights = F.softmax(torch.as_tensor([a_s, a_h]), dim=0)
#         elif self.curriculum_type == "hard":
#             weights = [0]
#         elif self.curriculum_type == "soft":
#             weights = [1]
#         self.soft_neg_weight = weights[0]
#         self.hard_neg_weight = 1-weights[0]
        
#     def mix_step(self):
#         # if self.milestone_updater.warmup_steps > 0: 
#         #     return f"warmup({self.milestone_updater.warmup_steps}) {self.milestone_updater.mixing_rate}({self.milestone_updater.steps[self.milestone_updater.i]}) "
#         # else: return f"mix: {self.milestone_updater.mixing_rate}({self.milestone_updater.steps[self.milestone_updater.i]}) "
#         if self.curriculum_type == "mr":
#             if self.warmup_steps > 0:
#                 return f"w({self.warmup_steps}) "
#             else: return f"{self.soft_neg_weight:.3f}|{self.hard_neg_weight:.3f} "
#         elif self.curriculum_type == "lp":
#             return f"{self.soft_neg_weight:.3f}|{self.hard_neg_weight:.3f} s:{self.lp_s:.3f}|h:{self.lp_h:.3f} "
#         elif self.curriculum_type == "exp":
#             r = self.step_ctr/self.Z
#             return f"{self.soft_neg_weight:.3f}|{self.hard_neg_weight:.3f} r:{r:.3f} "
#         elif self.curriculum_type == "hard":
#             return ""
        
#     def __len__(self):
#         return len(self.data)
    
#     def _proc_text(self, text: str) -> str:
#         text = " ".join(text.split("\n"))
#         text = " ".join(text.split()).strip()
#         return text
    
#     def _proc_code(self, code: str) -> Union[str, Tuple[str, list]]:
#         """returns processed code for CodeBERT and UniXcoder,
#         returns proccessed code and dataflow graph for GraphCodeBERT."""
#         if self.model_name == "graphcodebert":
#             return self._graphcodebert_proc_code(code)
#         else:
#             return self._unxicoder_codebert_proc_code(code)
        
#     def _unxicoder_codebert_proc_code(self, code: str):
#         code = " ".join(code.split("\n")).strip()
#         return code
    
#     def _graphcodebert_proc_code(self, code: str):
#         from datautils.parser import (remove_comments_and_docstrings, tree_to_token_index, 
#                                       index_to_code_token, tree_to_variable_index)
#         try: code = remove_comments_and_docstrings(code, 'python')
#         except: pass
#         tree = self.parser[0].parse(bytes(code, 'utf8'))    
#         root_node = tree.root_node  
#         tokens_index = tree_to_token_index(root_node)     
#         code = code.split('\n')
#         code_tokens = [index_to_code_token(x,code) for x in tokens_index]  
#         index_to_code = {}
#         for idx,(index,code) in enumerate(zip(tokens_index, code_tokens)):
#             index_to_code[index]=(idx, code)  
#         try: DFG,_ = self.parser[1](root_node, index_to_code, {}) 
#         except Exception as e:
#             print("Ln 380:", e)
#             DFG = []
#         DFG = sorted(DFG, key=lambda x: x[1])
#         indexs = set()
#         for d in DFG:
#             if len(d[-1]) != 0: indexs.add(d[1])
#             for x in d[-1]: indexs.add(x)
#         new_DFG = []
#         for d in DFG:
#             if d[1] in indexs: new_DFG.append(d)
#         dfg = new_DFG 
        
#         return code_tokens, dfg
        
#     def _retrieve_best_triplet(self, NL: str, PL: str, use_AST: bool, 
#                                batch_size: int=48, stochastic=True,
#                                backup_neg: Union[str, None]=None):
#         rindex = 0
#         codes_for_sim_intents: List[str] = []
#         rules_for_sim_intents: List[int] = []
#         if use_AST: # when using AST only use AST.
#             for tup in self.perturbed_codes[PL]: # codes from AST.
#                 if tup[1] in IGNORE_LIST: continue
#                 rule_index = int(tup[1].replace("rule",""))
#                 codes_for_sim_intents.append(tup[0])
#                 rules_for_sim_intents.append(rule_index)
#             # print(codes_for_sim_intents)
#         else: # TODO: add a flag for IDNS.
#             sim_intents: List[str] = self.sim_intents_map[NL]
#             for intent, _ in sim_intents:
#                 codes_for_sim_intents += self.intent_to_code[intent]
#                 rules_for_sim_intents += [-1]*len(self.intent_to_code[intent])
#         # print("codes_for_sim_intents:", codes_for_sim_intents)
#         if len(codes_for_sim_intents) == 0: # if no pool of backup candidates is available.
#             neg = backup_neg
#         else:
#             self.model.eval()
#             # if len(codes_for_sim_intents) == 1: 
#             #     codes_for_sim_intents += backup_neg
#             #     rules_for_sim_intents += 0
#             with torch.no_grad():
#                 enc_text = torch.stack(self.model.encode_emb([NL], mode="text", 
#                                                              batch_size=batch_size,
#                                                              device_id=self.device)) # 1 x hidden_size
#                 enc_codes = torch.stack(self.model.encode_emb(
#                     codes_for_sim_intents, mode="code", 
#                     device_id=self.device, batch_size=batch_size
#                 )) # num_cands x hidden_size
#                 scores = enc_text @ enc_codes.T # 1 x num_cands
#             if stochastic:
#                 p = F.softmax(self.beta*scores.squeeze(), dim=0).cpu().numpy()
#                 try: i: int = np.random.choice(range(len(p)), p=p)
#                 except TypeError:
#                     return NL, PL, backup_neg, rindex
#             else:
#                 i: int = torch.topk(scores, k=1).indices[0].item()
#             msg = f"{len(rules_for_sim_intents)} rules != {len(codes_for_sim_intents)} codes"
#             assert len(rules_for_sim_intents) == len(codes_for_sim_intents), msg
#             neg = codes_for_sim_intents[i]
#             rindex = rules_for_sim_intents[i]
            
#         return NL, PL, neg, rindex
    
#     def _sample_rand_triplet(self, NL: str, PL: str):
#         codes = []
#         for intent in self.intent_to_code:
#             if intent != NL: 
#                 codes += self.intent_to_code[intent]
                
#         return NL, PL, random.choice(codes)
        
#     def __getitem__(self, item: int):
#         # combined get item for all 3 models: CodeBERT, GraphCodeBERT, UniXcoder.
#         if self.val or self.warmup_steps > 0:
#             hard_neg = 0
#         else:
#             hard_neg = np.random.choice(
#                 [0, 1], p=[
#                     self.soft_neg_weight, 
#                     self.hard_neg_weight,
#                 ])
#         # if curriculum is turned off then just use hard negatives all the time.
#         if not self.use_curriculum: hard_neg = 1
#         if self.curriculum_type == "rand":
#             hard_neg = np.random.choice([0, 1], p=[0.5, 0.5])
#         if hard_neg: # sample hard similar intent or AST based negatives.
#             # anchor, pos, neg = self._retrieve_best_triplet(
#             #     NL=self.data[item]["intent"], 
#             #     PL=self.data[item]["snippet"],
#             #     use_AST=self.use_AST,
#             # )
#             anchor, pos, neg, hard_neg = self._retrieve_best_triplet(
#                 NL=self.data[item][0], PL=self.data[item][1],
#                 use_AST=self.use_AST, backup_neg=self.data[item][2], # required if no AST based candidates available.
#             )
#         elif self.val: # a soft negative but during val step. No further sampling is needed here.
#             anchor = self.data[item][0]
#             pos = self.data[item][1]
#             neg = self.data[item][2]
#         else: # sample soft random negatives, during train step.
#             # anchor, pos, neg = self._sample_rand_triplet(
#             #     NL=self.data[item]["intent"], 
#             #     PL=self.data[item]["snippet"],
#             # )
#             anchor = self.data[item][0]
#             pos = self.data[item][1]
#             neg = self.data[item][2]
#         anchor = self._proc_text(anchor)
#         pos = self._proc_code(pos)
#         neg = self._proc_code(neg)
#         if self.model_name == "codebert":
#             return self._codebert_getitem(anchor, pos, neg, hard_neg)
#         elif self.model_name == "graphcodebert":
#             return self._graphcodebert_getitem(anchor, pos, neg, hard_neg)
#         elif self.model_name == "unixcoder":
#             return self._unixcoder_getitem(anchor, pos, neg, hard_neg)
        
#     def _codebert_getitem(self, anchor: str, pos: str, neg: str, hard_neg: bool):
#         # special tokens are added by default.
#         anchor = self.tokenizer(anchor, **self.tok_args)
#         pos = self.tokenizer(pos, **self.tok_args)
#         neg = self.tokenizer(neg, **self.tok_args)
#         return [
#             anchor["input_ids"][0], anchor["attention_mask"][0], 
#             pos["input_ids"][0], pos["attention_mask"][0],
#             neg["input_ids"][0], neg["attention_mask"][0],
#             torch.tensor(hard_neg),
#         ]
        
#     def _unixcoder_getitem(self, anchor: str, pos: str, neg: str, hard_neg: bool):
#         # special tokens are added by default.
#         anchor = self.model.embed_model.tokenize([anchor], **self.tok_args)[0]
#         pos = self.model.embed_model.tokenize([pos], **self.tok_args)[0]
#         neg = self.model.embed_model.tokenize([neg], **self.tok_args)[0]
#         # print(anchor)
#         return [torch.tensor(anchor), 
#                 torch.tensor(pos), 
#                 torch.tensor(neg),
#                 torch.tensor(hard_neg)] 
        
#     def _graphcodebert_getitem(self, anchor: str, pos: Union[str, list], neg: Union[str, list], hard_neg: bool):
#         args = self.tok_args
#         tokenizer = self.tokenizer
#         # nl
#         nl=anchor
#         nl_tokens=tokenizer.tokenize(nl)[:args["nl_length"]-2]
#         nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
#         nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
#         padding_length = args["nl_length"] - len(nl_ids)
#         nl_ids+=[tokenizer.pad_token_id]*padding_length 
#         # pos
#         code_tokens, dfg = pos
#         code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
#         ori2cur_pos={}
#         ori2cur_pos[-1]=(0,0)
#         for i in range(len(code_tokens)):
#             ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
#         code_tokens=[y for x in code_tokens for y in x]  
#         # truncating
#         code_tokens=code_tokens[:args["code_length"]+args["data_flow_length"]-2-min(len(dfg),args["data_flow_length"])]
#         code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
#         pos_code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
#         pos_position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
#         dfg=dfg[:args["code_length"]+args["data_flow_length"]-len(code_tokens)]
#         code_tokens+=[x[0] for x in dfg]
#         pos_position_idx+=[0 for x in dfg]
#         pos_code_ids+=[tokenizer.unk_token_id for x in dfg]
#         padding_length=args["code_length"]+args["data_flow_length"]-len(pos_code_ids)
#         pos_position_idx+=[tokenizer.pad_token_id]*padding_length
#         pos_code_ids+=[tokenizer.pad_token_id]*padding_length    
#         # reindex
#         reverse_index={}
#         for idx,x in enumerate(dfg):
#             reverse_index[x[1]]=idx
#         for idx,x in enumerate(dfg):
#             dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
#         dfg_to_dfg=[x[-1] for x in dfg]
#         dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
#         length=len([tokenizer.cls_token])
#         dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code] 
#         # calculate graph-guided masked function
#         pos_attn_mask=np.zeros((args["code_length"]+args["data_flow_length"],
#                                 args["code_length"]+args["data_flow_length"]),
#                                 dtype=bool)
#         # calculate begin index of node and max length of input
#         node_index=sum([i>1 for i in pos_position_idx])
#         max_length=sum([i!=1 for i in pos_position_idx])
#         # sequence can attend to sequence
#         pos_attn_mask[:node_index,:node_index]=True
#         # special tokens attend to all tokens
#         for idx,i in enumerate(pos_code_ids):
#             if i in [0,2]:
#                 pos_attn_mask[idx,:max_length]=True
#         # nodes attend to code tokens that are identified from
#         for idx,(a,b) in enumerate(dfg_to_code):
#             if a<node_index and b<node_index:
#                 pos_attn_mask[idx+node_index,a:b]=True
#                 pos_attn_mask[a:b,idx+node_index]=True
#         # nodes attend to adjacent nodes 
#         for idx,nodes in enumerate(dfg_to_dfg):
#             for a in nodes:
#                 if a+node_index<len(pos_position_idx):
#                     pos_attn_mask[idx+node_index,a+node_index]=True

#         # neg
#         code_tokens, dfg = neg
#         code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
#         ori2cur_pos={}
#         ori2cur_pos[-1]=(0,0)
#         for i in range(len(code_tokens)):
#             ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
#         code_tokens=[y for x in code_tokens for y in x]  
#         # truncating
#         code_tokens=code_tokens[:args["code_length"]+args["data_flow_length"]-2-min(len(dfg),args["data_flow_length"])]
#         code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
#         neg_code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
#         neg_position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
#         dfg=dfg[:args["code_length"]+args["data_flow_length"]
#                 -len(code_tokens)]
#         code_tokens+=[x[0] for x in dfg]
#         neg_position_idx+=[0 for x in dfg]
#         neg_code_ids+=[tokenizer.unk_token_id for x in dfg]
#         padding_length=args["code_length"]+args["data_flow_length"]-len(neg_code_ids)
#         neg_position_idx+=[tokenizer.pad_token_id]*padding_length
#         neg_code_ids+=[tokenizer.pad_token_id]*padding_length    
#         # reindex
#         reverse_index={}
#         for idx,x in enumerate(dfg):
#             reverse_index[x[1]]=idx
#         for idx,x in enumerate(dfg):
#             dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
#         dfg_to_dfg=[x[-1] for x in dfg]
#         dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
#         length=len([tokenizer.cls_token])
#         dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code] 

#         # calculate graph-guided masked function
#         neg_attn_mask=np.zeros((args["code_length"]+args["data_flow_length"],
#                             args["code_length"]+args["data_flow_length"]),dtype=bool)
#         # calculate begin index of node and max length of input
#         node_index=sum([i>1 for i in neg_position_idx])
#         max_length=sum([i!=1 for i in neg_position_idx])
#         # sequence can attend to sequence
#         neg_attn_mask[:node_index,:node_index]=True
#         # special tokens attend to all tokens
#         for idx,i in enumerate(neg_code_ids):
#             if i in [0,2]:
#                 neg_attn_mask[idx,:max_length]=True
#         # nodes attend to code tokens that are identified from
#         for idx,(a,b) in enumerate(dfg_to_code):
#             if a<node_index and b<node_index:
#                 neg_attn_mask[idx+node_index,a:b]=True
#                 neg_attn_mask[a:b,idx+node_index]=True
#         # nodes attend to adjacent nodes 
#         for idx,nodes in enumerate(dfg_to_dfg):
#             for a in nodes:
#                 if a+node_index<len(neg_position_idx):
#                     neg_attn_mask[idx+node_index,a+node_index]=True

#         return (
#                 torch.tensor(pos_code_ids),
#                 torch.tensor(pos_attn_mask),
#                 torch.tensor(pos_position_idx),
#                 torch.tensor(neg_code_ids),
#                 torch.tensor(neg_attn_mask),
#                 torch.tensor(neg_position_idx),
#                 torch.tensor(nl_ids),
#                 torch.tensor(hard_neg),
#                )
# NL-PL pairs dataset class.
class AllModelsDataset(Dataset):
    def __init__(self, path: str, model_name: str, tokenizer=None, 
                 ignore_new_rules: bool=False, ignore_worst_rules: bool=False,
                 ignore_non_disco_rules: bool=False, ignore_old_worst_rules: bool=False,
                 ignore_unnatural_rules: bool=False, **tok_args):
        super(AllModelsDataset, self).__init__()
        assert model_name in MODEL_OPTIONS
        # if filename endswith jsonl:
        self.ignore_new_rules = ignore_new_rules
        self.ignore_worst_rules = ignore_worst_rules
        self.ignore_old_worst_rules = ignore_old_worst_rules
        self.ignore_unnatural_rules = ignore_unnatural_rules
        self.ignore_non_disco_rules = ignore_non_disco_rules
        if path.endswith(".jsonl"):
            self.data = read_jsonl(path) # NL-PL pairs.
        # if filename endswith json:
        elif path.endswith(".json"):
            self.data = json.load(open(path)) # NL-PL pairs.
        # if filename endswith jsonl:
        if path.endswith(".jsonl"):
            self.data = read_jsonl(path) # NL-PL pairs.
        # if filename endswith json:
        elif path.endswith(".json"):
            self.data = json.load(open(path)) # NL-PL pairs.
        # parser is needed for GraphCodeBERT to get the dataflow.
        if model_name == "graphcodebert":
            from datautils.parser import DFG_python
            from tree_sitter import Language, Parser
            PARSER =  Parser()
            LANGUAGE = Language('datautils/py_parser.so', 'python')
            PARSER.set_language(LANGUAGE)
            self.parser = [PARSER, DFG_python]
        self.model_name = model_name
        self.tok_args = tok_args
        if isinstance(tokenizer, RobertaTokenizer): 
            self.tokenizer = tokenizer
        elif isinstance(tokenizer, str):
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
        else: self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def _proc_text(self, text: str) -> str:
        text = " ".join(text.split("\n"))
        text = " ".join(text.split()).strip()
        return text
    
    def _proc_code(self, code: str) -> Union[str, Tuple[str, list]]:
        """returns processed code for CodeBERT and UniXcoder,
        returns proccessed code and dataflow graph for GraphCodeBERT."""
        if self.model_name == "graphcodebert":
            return self._graphcodebert_proc_code(code)
        else:
            return self._unxicoder_codebert_proc_code(code)
        
    def _unxicoder_codebert_proc_code(self, code: str):
        code = " ".join(code.split("\n")).strip()
        return code
    
    def _graphcodebert_proc_code(self, code: str):
        from datautils.parser import (remove_comments_and_docstrings, tree_to_token_index, 
                                      index_to_code_token, tree_to_variable_index)
        try: code = remove_comments_and_docstrings(code, 'python')
        except: pass
        tree = self.parser[0].parse(bytes(code, 'utf8'))    
        root_node = tree.root_node  
        tokens_index = tree_to_token_index(root_node)     
        code = code.split('\n')
        code_tokens = [index_to_code_token(x,code) for x in tokens_index]  
        index_to_code = {}
        for idx,(index,code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index]=(idx, code)  
        try: DFG,_ = self.parser[1](root_node, index_to_code, {}) 
        except Exception as e:
            print("Ln 380:", e)
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0: indexs.add(d[1])
            for x in d[-1]: indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs: new_DFG.append(d)
        dfg = new_DFG 
        
        return code_tokens, dfg
    
    def _graphcodebert_proc_text(self, nl: str):
        nl_tokens = self.tokenizer.tokenize(nl)[:self.tok_args["nl_length"]-2]
        nl_tokens = [self.tokenizer.cls_token]+nl_tokens+[self.tokenizer.sep_token]
        nl_ids = self.tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = self.tok_args["nl_length"] - len(nl_ids)
        nl_ids += [self.tokenizer.pad_token_id]*padding_length 
        
        return nl_ids
        
    def _retrieve_best_triplet(self, NL: str, PL: str, use_AST: bool, 
                               batch_size: int=48, stochastic=True,
                               backup_neg: Union[str, None]=None):
        rindex = 0
        codes_for_sim_intents: List[str] = []
        rules_for_sim_intents: List[int] = []
        if use_AST: # when using AST only use AST.
            for tup in self.perturbed_codes[PL]: # codes from AST.
                if self.ignore_new_rules and tup[1] in NEW_RULES_IGNORE_LIST: continue
                elif self.ignore_worst_rules and tup[1] in WORST_RULES_LIST: continue
                elif self.ignore_non_disco_rules and tup[1] in DISCO_IGNORE_LIST: continue
                elif self.ignore_old_worst_rules and tup[1] in WORST_OLD_RULES_LIST: continue
                elif self.ignore_unnatural_rules and tup[1] in UNNATURAL_IGNORE_LIST: continue
                rule_index = int(tup[1].replace("rule",""))
                codes_for_sim_intents.append(tup[0])
                rules_for_sim_intents.append(rule_index)
            # print(codes_for_sim_intents)
        else: # TODO: add a flag for IDNS.
            sim_intents: List[str] = self.sim_intents_map[NL]
            for intent, _ in sim_intents:
                codes_for_sim_intents += self.intent_to_code[intent]
                rules_for_sim_intents += [-1]*len(self.intent_to_code[intent])
        # print("codes_for_sim_intents:", codes_for_sim_intents)
        if len(codes_for_sim_intents) == 0: # if no pool of backup candidates is available.
            neg = backup_neg
        else:
            self.model.eval()
            # if len(codes_for_sim_intents) == 1: 
            #     codes_for_sim_intents += backup_neg
            #     rules_for_sim_intents += 0
            with torch.no_grad():
                enc_text = torch.stack(self.model.encode_emb([NL], mode="text", 
                                                             batch_size=batch_size,
                                                             device_id=self.device)) # 1 x hidden_size
                enc_codes = torch.stack(self.model.encode_emb(
                    codes_for_sim_intents, mode="code", 
                    device_id=self.device, batch_size=batch_size
                )) # num_cands x hidden_size
                scores = enc_text @ enc_codes.T # 1 x num_cands
            if stochastic:
                p = F.softmax(self.beta*scores.squeeze(), dim=0).cpu().numpy()
                try: i: int = np.random.choice(range(len(p)), p=p)
                except TypeError:
                    return NL, PL, backup_neg, rindex
            else:
                i: int = torch.topk(scores, k=1).indices[0].item()
            msg = f"{len(rules_for_sim_intents)} rules != {len(codes_for_sim_intents)} codes"
            assert len(rules_for_sim_intents) == len(codes_for_sim_intents), msg
            neg = codes_for_sim_intents[i]
            rindex = rules_for_sim_intents[i]
            
        return NL, PL, neg, rindex

    def _codebert_getitem(self, anchor: str, pos: str, neg: str, hard_neg: bool):
        # special tokens are added by default.
        anchor = self.tokenizer(anchor, **self.tok_args)
        pos = self.tokenizer(pos, **self.tok_args)
        neg = self.tokenizer(neg, **self.tok_args)
        return [
            anchor["input_ids"][0], anchor["attention_mask"][0], 
            pos["input_ids"][0], pos["attention_mask"][0],
            neg["input_ids"][0], neg["attention_mask"][0],
            torch.tensor(hard_neg),
        ]
        
    def _unixcoder_getitem(self, anchor: str, pos: str, neg: str, hard_neg: bool):
        # special tokens are added by default.
        anchor = self.tokenizer([anchor], **self.tok_args)[0]
        pos = self.tokenizer([pos], **self.tok_args)[0]
        neg = self.tokenizer([neg], **self.tok_args)[0]
        # print(anchor)
        return [torch.tensor(anchor), 
                torch.tensor(pos), 
                torch.tensor(neg),
                torch.tensor(hard_neg)] 
        
    def _graphcodebert_text_encode(self, nl: str):
        nl_tokens=self.tokenizer.tokenize(nl)[:self.tok_args["nl_length"]-2]
        nl_tokens =[self.tokenizer.cls_token]+nl_tokens+[self.tokenizer.sep_token]
        nl_ids =  self.tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = self.tok_args["nl_length"] - len(nl_ids)
        nl_ids+=[self.tokenizer.pad_token_id]*padding_length 
        
        return nl_ids

    def _graphcodebert_code_encode(self, code_and_dfg: tuple):
        # pos
        code_tokens, dfg = code_and_dfg
        code_tokens=[self.tokenizer.tokenize('@ '+x)[1:] if idx!=0 else self.tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
        ori2cur_pos={}
        ori2cur_pos[-1]=(0,0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
        code_tokens=[y for x in code_tokens for y in x]  
        # truncating
        code_tokens=code_tokens[:self.tok_args["code_length"]+self.tok_args["data_flow_length"]-2-min(len(dfg),self.tok_args["data_flow_length"])]
        code_tokens =[self.tokenizer.cls_token]+code_tokens+[self.tokenizer.sep_token]
        code_ids =  self.tokenizer.convert_tokens_to_ids(code_tokens)
        position_idx = [i+self.tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
        dfg=dfg[:self.tok_args["code_length"]+self.tok_args["data_flow_length"]-len(code_tokens)]
        code_tokens+=[x[0] for x in dfg]
        position_idx+=[0 for x in dfg]
        code_ids+=[self.tokenizer.unk_token_id for x in dfg]
        padding_length=self.tok_args["code_length"]+self.tok_args["data_flow_length"]-len(code_ids)
        position_idx+=[self.tokenizer.pad_token_id]*padding_length
        code_ids+=[self.tokenizer.pad_token_id]*padding_length    
        # reindex
        reverse_index={}
        for idx,x in enumerate(dfg):
            reverse_index[x[1]]=idx
        for idx,x in enumerate(dfg):
            dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
        dfg_to_dfg=[x[-1] for x in dfg]
        dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
        length=len([self.tokenizer.cls_token])
        dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code] 
        # calculate graph-guided masked function
        attn_mask=np.zeros((self.tok_args["code_length"]+self.tok_args["data_flow_length"],
                                self.tok_args["code_length"]+self.tok_args["data_flow_length"]),
                                dtype=bool)
        # calculate begin index of node and max length of input
        node_index=sum([i>1 for i in position_idx])
        max_length=sum([i!=1 for i in position_idx])
        # sequence can attend to sequence
        attn_mask[:node_index,:node_index]=True
        # special tokens attend to all tokens
        for idx,i in enumerate(code_ids):
            if i in [0,2]:
                attn_mask[idx,:max_length]=True
        # nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(dfg_to_code):
            if a<node_index and b<node_index:
                attn_mask[idx+node_index,a:b]=True
                attn_mask[a:b,idx+node_index]=True
        # nodes attend to adjacent nodes 
        for idx,nodes in enumerate(dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(position_idx):
                    attn_mask[idx+node_index,a+node_index]=True

        return code_ids, attn_mask, position_idx
        
    def _graphcodebert_getitem(self, anchor: str, pos: Union[str, list], neg: Union[str, list], hard_neg: bool):
        nl_ids = self._graphcodebert_proc_text(nl=anchor) # nl
        pos_code_ids, pos_attn_mask, pos_position_idx = self._graphcodebert_code_encode(code_and_dfg=pos) # pos
        neg_code_ids, neg_attn_mask, neg_position_idx = self._graphcodebert_code_encode(code_and_dfg=neg) # neg
        
        return (
                torch.tensor(pos_code_ids),
                torch.tensor(pos_attn_mask),
                torch.tensor(pos_position_idx),
                torch.tensor(neg_code_ids),
                torch.tensor(neg_attn_mask),
                torch.tensor(neg_position_idx),
                torch.tensor(nl_ids),
                torch.tensor(hard_neg),
               )

# Dataset for dynamic creation of triples.
class DynamicTriplesDataset(AllModelsDataset):
    def __init__(self, path: str, model_name: str, model=None, tokenizer=None,
                 use_AST=False, val=False, warmup_steps=3000, beta=0.001, p=2,
                 sim_intents_map={}, perturbed_codes={}, device="cuda:0", win_size=20, 
                 delta=0.5, epsilon=0.8, use_curriculum=True, curriculum_type="mr", 
                 soft_neg_bias=0.8, batch_size=None, num_epochs=None, **tok_args):
        super(DynamicTriplesDataset, self).__init__(
            path=path, model_name=model_name,
            tokenizer=tokenizer, **tok_args,
        )
        self.use_curriculum = use_curriculum
        # print(f"\x1b[31;1m{curriculum_type}\x1b[0m")
        # self.rand_curriculum = rand_curriculum
        # check if valid curriculum type:
        msg = f"invalid curriculum type: {curriculum_type}"
        assert curriculum_type in ["mr", "rand", "lp", 
                                   "exp", "hard", "soft"], msg
        self.curriculum_type = curriculum_type
        if curriculum_type != "mr": 
            warmup_steps = 0
        self.step_ctr = 0
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        # self.milestone_updater = MilestoneUpdater()
        self.p = p
        self.beta = beta
        self.epsilon = epsilon
        self.soft_master_rate = MasteringRate(
            role=0, delta=delta,
            p=p, size=win_size, 
        )
        self.hard_master_rate = MasteringRate(
            role=1, delta=delta,
            p=p, size=win_size, 
        )
        self.soft_neg_bias = soft_neg_bias
        self.soft_neg_weight = 0.8
        self.hard_neg_weight = 0.2
        if self.curriculum_type == "hard":
            self.soft_neg_weight = 0
            self.hard_neg_weight = 1
        elif self.curriculum_type == "soft":
            self.soft_neg_weight = 1
            self.hard_neg_weight = 0
        self.model = model # pointer to model instance to find closest NL & PL examples
        self.sim_intents_map = sim_intents_map
        self.perturbed_codes = perturbed_codes
        self.use_AST = use_AST
        self.device = device
        self.val = val
        self.lp_s = 0
        self.lp_h = 0
        # create a mapping of NL to all associated PLs. 
        self.intent_to_code = {}
        for rec in self.data:
            try: intent = rec["intent"]
            except TypeError: intent = rec[0]
            try: snippet = rec["snippet"]
            except TypeError: snippet = rec[1]
            try: self.intent_to_code[intent].append(snippet)
            except KeyError: self.intent_to_code[intent] = [snippet]
        if curriculum_type == "exp":
            assert batch_size is not None, "need batch size for exponential decay curriculum"
            assert num_epochs is not None, "need num epochs for exponential decay curriculum"
            N = len(self)*self.num_epochs
            half_life_frac = 0.8 # this means the weights will be halved when 50% of training is complete
            self.Z = N*half_life_frac
            print("using exponentialy decaying curriculum")
        
    def update(self, soft_acc: float, hard_acc: float):
        # self.milestone_updater.update(acc)
        self.soft_master_rate.push_acc(soft_acc)
        self.hard_master_rate.push_acc(hard_acc)
        if self.curriculum_type == "mr":
            if self.warmup_steps > 0: self.warmup_steps -= 1; return 
            a_s = self.soft_master_rate.attn(self.hard_master_rate)
            a_h = self.hard_master_rate.attn(self.soft_master_rate)
            attn_dist = F.softmax(torch.as_tensor([a_s, a_h]), dim=0)
            bias_dist = torch.as_tensor([self.soft_neg_bias, 1-self.soft_neg_bias])
            weights = (1-self.epsilon)*attn_dist + self.epsilon*bias_dist
            weights = weights.numpy()
        elif self.curriculum_type == "exp":
            self.step_ctr += self.batch_size
            r = self.step_ctr/self.Z
            weights = [np.exp(-np.log(2)*r)]
        elif self.curriculum_type == "lp":
            lp_s = self.soft_master_rate()
            lp_h = self.hard_master_rate()
            a_s = (1-lp_s)*(1-lp_h)
            a_h = (lp_s**self.p)*(1-lp_h)
            self.lp_s, self.lp_h = lp_s, lp_h
            weights = F.softmax(torch.as_tensor([a_s, a_h]), dim=0)
        elif self.curriculum_type == "hard":
            weights = [0]
        elif self.curriculum_type == "soft":
            weights = [1]
        self.soft_neg_weight = weights[0]
        self.hard_neg_weight = 1-weights[0]
        
    def mix_step(self):
        # if self.milestone_updater.warmup_steps > 0: 
        #     return f"warmup({self.milestone_updater.warmup_steps}) {self.milestone_updater.mixing_rate}({self.milestone_updater.steps[self.milestone_updater.i]}) "
        # else: return f"mix: {self.milestone_updater.mixing_rate}({self.milestone_updater.steps[self.milestone_updater.i]}) "
        if self.curriculum_type == "mr":
            if self.warmup_steps > 0:
                return f"w({self.warmup_steps}) "
            else: return f"{self.soft_neg_weight:.3f}|{self.hard_neg_weight:.3f} "
        elif self.curriculum_type == "lp":
            return f"{self.soft_neg_weight:.3f}|{self.hard_neg_weight:.3f} s:{self.lp_s:.3f}|h:{self.lp_h:.3f} "
        elif self.curriculum_type == "exp":
            r = self.step_ctr/self.Z
            return f"{self.soft_neg_weight:.3f}|{self.hard_neg_weight:.3f} r:{r:.3f} "
        elif self.curriculum_type == "hard":
            return ""
    
    def _sample_rand_triplet(self, NL: str, PL: str):
        codes = []
        for intent in self.intent_to_code:
            if intent != NL: 
                codes += self.intent_to_code[intent]
                
        return NL, PL, random.choice(codes)
        
    def __getitem__(self, item: int):
        # combined get item for all 3 models: CodeBERT, GraphCodeBERT, UniXcoder.
        if self.val or self.warmup_steps > 0:
            hard_neg = 0
        else:
            hard_neg = np.random.choice(
                [0, 1], p=[
                    self.soft_neg_weight, 
                    self.hard_neg_weight,
                ])
        # if curriculum is turned off then just use hard negatives all the time.
        if not self.use_curriculum: hard_neg = 1
        if self.curriculum_type == "rand":
            hard_neg = np.random.choice([0, 1], p=[0.5, 0.5])
        if hard_neg: # sample hard similar intent or AST based negatives.
            # anchor, pos, neg = self._retrieve_best_triplet(
            #     NL=self.data[item]["intent"], 
            #     PL=self.data[item]["snippet"],
            #     use_AST=self.use_AST,
            # )
            anchor, pos, neg, hard_neg = self._retrieve_best_triplet(
                NL=self.data[item][0], PL=self.data[item][1],
                use_AST=self.use_AST, backup_neg=self.data[item][2], # required if no AST based candidates available.
            )
        elif self.val: # a soft negative but during val step. No further sampling is needed here.
            anchor = self.data[item][0]
            pos = self.data[item][1]
            neg = self.data[item][2]
        else: # sample soft random negatives, during train step.
            # anchor, pos, neg = self._sample_rand_triplet(
            #     NL=self.data[item]["intent"], 
            #     PL=self.data[item]["snippet"],
            # )
            anchor = self.data[item][0]
            pos = self.data[item][1]
            neg = self.data[item][2]
        anchor = self._proc_text(anchor)
        pos = self._proc_code(pos)
        neg = self._proc_code(neg)
        if self.model_name == "codebert":
            return self._codebert_getitem(anchor, pos, neg, hard_neg)
        elif self.model_name == "graphcodebert":
            return self._graphcodebert_getitem(anchor, pos, neg, hard_neg)
        elif self.model_name == "unixcoder":
            return self._unixcoder_getitem(anchor, pos, neg, hard_neg)

# Retrieval based validation.
class ValRetDataset(Dataset):
    """JUST a convenience class to convert NL-PL pairs to retrieval setting."""
    def __init__(self, path: str):
        super(ValRetDataset, self).__init__()
        self.data = json.load(open(path))
        posts = {} # query to candidate map.
        cands = {} # unique candidates
        tot = 0
        for rec in self.data:
            intent = rec["intent"]
            snippet = rec["snippet"]
            if snippet not in cands:
                cands[snippet] = tot
                tot += 1
            try: posts[intent][snippet] = None
            except KeyError: posts[intent] = {snippet: None}
        self._cands = list(cands.keys())
        self._queries = list(posts.keys())
        self._labels = []
        for query, candidates in posts.items():
            self._labels.append([])
            for cand in candidates.keys():
                self._labels[-1].append(cands[cand])

    def __len__(self):
        return len(self.data)
    
    def get_queries(self):
        return self._queries
    
    def get_candidates(self):
        return self._cands
    
    def get_labels(self):
        return self._labels
        
    def __getitem__(self, i: int):
        return self.data[i]
    
# QuadruplesDataset: (NL, PL, soft_neg PL, hard_neg PL)
class QuadruplesDataset(AllModelsDataset):
    def __init__(self, path: str, model_name: str, model=None, tokenizer=None,
                 use_AST=False, val=False, beta=0.001, sim_intents_map={}, 
                 perturbed_codes={}, device="cuda:0", batch_size=None, 
                 num_epochs=None, **tok_args):
        super(QuadruplesDataset, self).__init__(
            path=path, model_name=model_name,
            tokenizer=tokenizer, **tok_args,
        )
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.beta = beta
        self.model = model # pointer to model instance to find closest NL & PL examples
        self.sim_intents_map = sim_intents_map
        self.perturbed_codes = perturbed_codes
        self.use_AST = use_AST
        self.device = device
        self.val = val
        # create a mapping of NL to all associated PLs. 
        self.intent_to_code = {}
        for rec in self.data:
            try: intent = rec["intent"]
            except TypeError: intent = rec[0]
            try: snippet = rec["snippet"]
            except TypeError: snippet = rec[1]
            try: self.intent_to_code[intent].append(snippet)
            except KeyError: self.intent_to_code[intent] = [snippet]
        
    def mix_step(self):
        # if self.milestone_updater.warmup_steps > 0: 
        #     return f"warmup({self.milestone_updater.warmup_steps}) {self.milestone_updater.mixing_rate}({self.milestone_updater.steps[self.milestone_updater.i]}) "
        # else: return f"mix: {self.milestone_updater.mixing_rate}({self.milestone_updater.steps[self.milestone_updater.i]}) "
        if self.curriculum_type == "mr":
            if self.warmup_steps > 0:
                return f"w({self.warmup_steps}) "
            else: return f"{self.soft_neg_weight:.3f}|{self.hard_neg_weight:.3f} "
        elif self.curriculum_type == "lp":
            return f"{self.soft_neg_weight:.3f}|{self.hard_neg_weight:.3f} s:{self.lp_s:.3f}|h:{self.lp_h:.3f} "
        elif self.curriculum_type == "exp":
            r = self.step_ctr/self.Z
            return f"{self.soft_neg_weight:.3f}|{self.hard_neg_weight:.3f} r:{r:.3f} "
        elif self.curriculum_type == "hard":
            return ""
        
    def _codebert_getitem(self, anchor: str, pos: str, 
                          soft_neg: str, hard_neg: str):
        # special tokens are added by default.
        anchor = self.tokenizer(anchor, **self.tok_args)[0]
        pos = self.tokenizer(pos, **self.tok_args)[0]
        soft_neg = self.tokenizer(neg, **self.tok_args)[0]
        hard_neg = self.tokenizer(neg, **self.tok_args)[0]
        return [
            anchor["input_ids"], anchor["attention_mask"], 
            pos["input_ids"], pos["attention_mask"],
            soft_neg["input_ids"], soft_neg["attention_mask"],
            hard_neg["input_ids"], hard_neg["attention_mask"]
        ]
        
    def _unixcoder_getitem(self, anchor: str, pos: str, 
                           soft_neg: str, hard_neg: str):
        # special tokens are added by default.
        anchor = self.model.embed_model.tokenize([anchor], **self.tok_args)[0]
        pos = self.model.embed_model.tokenize([pos], **self.tok_args)[0]
        soft_neg = self.model.embed_model.tokenize([neg], **self.tok_args)[0]
        hard_neg = self.model.embed_model.tokenize([neg], **self.tok_args)[0]
        # print(anchor)
        return [torch.tensor(anchor), 
                torch.tensor(pos), 
                torch.tensor(soft_neg),
                torch.tensor(hard_neg)] 
        
    def _graphcodebert_getitem(self, anchor: str, pos: Union[str, list], soft_neg, hard_neg):
        nl_ids = self._graphcodebert_proc_text(nl=anchor) # nl
        pos_code_ids, pos_attn_mask, pos_position_idx = self._graphcodebert_code_encode(code_and_dfg=pos) # pos
        soft_neg_code_ids, soft_neg_attn_mask, soft_neg_position_idx = self._graphcodebert_code_encode(code_and_dfg=soft_neg) # soft neg
        hard_neg_code_ids, hard_neg_attn_mask, hard_neg_position_idx = self._graphcodebert_code_encode(code_and_dfg=hard_neg) # hard neg
        
        return (
                torch.tensor(pos_code_ids),
                torch.tensor(pos_attn_mask),
                torch.tensor(pos_position_idx),
                torch.tensor(soft_neg_code_ids),
                torch.tensor(soft_neg_attn_mask),
                torch.tensor(soft_neg_position_idx),
                torch.tensor(hard_neg_code_ids),
                torch.tensor(hard_neg_attn_mask),
                torch.tensor(hard_neg_position_idx),
                torch.tensor(nl_ids),
               )
        
    def __getitem__(self, item: int):
        # combined get item for all 3 models: CodeBERT, GraphCodeBERT, UniXcoder.
        # if curriculum is turned off then just use hard negatives all the time.
        anchor = self.data[item][0]
        pos = self.data[item][1]
        neg = self.data[item][2]
        anchor = self._proc_text(anchor)
        pos = self._proc_code(pos)
        soft_neg = self._proc_code(soft_neg)
        hard_neg = self._proc_code(hard_neg)
        if self.model_name == "codebert":
            return self._codebert_getitem(anchor, pos, soft_neg, hard_neg)
        elif self.model_name == "graphcodebert":
            return self._graphcodebert_getitem(anchor, pos, soft_neg, hard_neg)
        elif self.model_name == "unixcoder":
            return self._unixcoder_getitem(anchor, pos, soft_neg, hard_neg)
    
def create_apn_from_ccp_ncp(data: List[dict], code_code_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
    """create A,P,N triplets from code-code pairs and nl-code pairs."""
    apn, ctr = [], 0
    c2c_map = defaultdict(lambda:[])
    for c1,c2 in code_code_pairs:
        c2c_map[c1].append(c2)
        c2c_map[c2].append(c1)
    c2c_map = dict(c2c_map)
    for rec in data:
        a = rec["intent"]
        p = rec["snippet"]
        sim_codes = c2c_map.get(p)
        if sim_codes is None:
            apn.append((a, p, p))
        else:
            n = random.sample(sim_codes, k=1)[0]
            # for n in sim_codes:
            apn.append((a, p, n))
    # print(ctr)
    return apn

def create_app_from_csyn_ncp(data: List[dict], code_synsets) -> List[Tuple[str, str, str]]:
    """create A,P,P+ triples from code-code synsets and nl-code pairs."""
    app, ctr = [], 0
    for rec in data:
        a = rec["intent"]
        p = rec["snippet"]
        p_ = code_synsets.pick_lex(p)
        app.append((a, p, p_))

    return app

# dataset class for CodeRetriever objective based training.
class CodeRetrieverDataset(AllModelsDataset):
    def __init__(self, nl_code_path: str, code_code_path: str, 
                 model_name: str, tokenizer=None, **tok_args):
        super(CodeRetrieverDataset, self).__init__(
            path=nl_code_path, model_name=model_name,
            tokenizer=tokenizer, **tok_args,
        )
        self.nl_code_path = nl_code_path
        self.code_code_path = code_code_path
        self.code_pairs = json.load(open(code_code_path))
        self.train_data = copy.deepcopy(self.data)
        self.data = create_apn_from_ccp_ncp(self.train_data, self.code_pairs)
        print(self.data[0])
        
    def reset(self):
        """reset code pairs"""
        self.data = create_apn_from_ccp_ncp(self.train_data, self.code_pairs)
        
    def __getitem__(self, item: int):
        """combined get item for all 3 models: CodeBERT, GraphCodeBERT, UniXcoder.
        if curriculum is turned off then just use hard negatives all the time."""
        anchor = self.data[item][0]
        pos = self.data[item][1]
        neg = self.data[item][2]
        anchor = self._proc_text(anchor)
        pos = self._proc_code(pos)
        neg = self._proc_code(neg)
        if self.model_name == "codebert":
            return self._codebert_getitem(anchor, pos, neg, False)
        elif self.model_name == "graphcodebert":
            return self._graphcodebert_getitem(anchor, pos, neg, False)
        elif self.model_name == "unixcoder":
            return self._unixcoder_getitem(anchor, pos, neg, False)  
        
# dataset class for CodeRetriever objective based training with triples and without unimodal loss.
class CodeRetrieverTriplesDataset(AllModelsDataset):
    def __init__(self, nl_code_path: str, model_name: str, 
                 tokenizer=None, **tok_args):
        super(CodeRetrieverTriplesDataset, self).__init__(
            path=nl_code_path, model_name=model_name,
            tokenizer=tokenizer, **tok_args,
        )
        self.nl_code_path = nl_code_path
        # create a mapping of NL to all associated PLs. 
        self.intent_to_code = {}
        for rec in self.data:
            try: intent = rec["intent"]
            except TypeError: intent = rec[0]
            try: snippet = rec["snippet"]
            except TypeError: snippet = rec[1]
            try: self.intent_to_code[intent].append(snippet)
            except KeyError: self.intent_to_code[intent] = [snippet]
        
    def reset(self): pass # just for API consistency

    def _sample_soft_neg(self, intent: str):
        """sample a soft negative: first sample a random intent then sample a random negative"""
        sampled_intent = intent
        while sampled_intent == intent:
            sampled_intent = random.sample(list(self.intent_to_code.keys()), k=1)[0]
        return random.sample(self.intent_to_code[sampled_intent], k=1)[0]
        
    def __getitem__(self, item: int):
        """combined get item for all 3 models: CodeBERT, GraphCodeBERT, UniXcoder.
        if curriculum is turned off then just use hard negatives all the time."""
        anchor = self.data[item]["intent"]
        pos = self.data[item]["snippet"]
        neg = self._sample_soft_neg(anchor)
        anchor = self._proc_text(anchor)
        pos = self._proc_code(pos)
        neg = self._proc_code(neg)
        if self.model_name == "codebert":
            return self._codebert_getitem(anchor, pos, neg, False)
        elif self.model_name == "graphcodebert":
            return self._graphcodebert_getitem(anchor, pos, neg, False)
        elif self.model_name == "unixcoder":
            return self._unixcoder_getitem(anchor, pos, neg, False)
        
# dataset class for CodeRetriever objective based training with quads and without unimodal loss.
class CodeRetrieverQuintsDataset(AllModelsDataset):
    def __init__(self, path: str, model_name: str, 
                 tokenizer=None, **tok_args):
        super(CodeRetrieverQuintsDataset, self).__init__(
            path=path, model_name=model_name,
            tokenizer=tokenizer, **tok_args,
        )
        # create a mapping of NL to all associated PLs. 
        self.intent_to_code = {}
        for rec in self.data:
            try: intent = rec["intent"]
            except TypeError: intent = rec[0]
            try: snippet = rec["snippet"]
            except TypeError: snippet = rec[1]
            try: self.intent_to_code[intent].append(snippet)
            except KeyError: self.intent_to_code[intent] = [snippet]
        
    def reset(self): pass # just for API consistency

    def _sample_soft_neg(self, intent: str):
        """sample a soft negative: first sample a random intent then sample a random negative"""
        sampled_intent = intent
        while sampled_intent == intent:
            sampled_intent = random.sample(list(self.intent_to_code.keys()), k=1)[0]
        return random.sample(self.intent_to_code[sampled_intent], k=1)[0]
                
    def _codebert_getitem(self, a, p, n1, n2, n3):
        # special tokens are added by default.
        a = self.tokenizer(a, **self.tok_args)
        p = self.tokenizer(p, **self.tok_args)
        n1 = self.tokenizer(n1, **self.tok_args)
        n2 = self.tokenizer(n2, **self.tok_args)
        n3 = self.tokenizer(n3, **self.tok_args)
        return [
            a["input_ids"][0], a["attention_mask"][0], 
            p["input_ids"][0], p["attention_mask"][0],
            n1["input_ids"][0], n1["attention_mask"][0],
            n2["input_ids"][0], n2["attention_mask"][0],
            n3["input_ids"][0], n3["attention_mask"][0],
        ]

    def _unixcoder_getitem(self, a, p, n1, n2, n3):
        # special tokens are added by default.
        a = self.tokenizer([a], **self.tok_args)[0]
        p = self.tokenizer([p], **self.tok_args)[0]
        n1 = self.tokenizer([n1], **self.tok_args)[0]
        n2 = self.tokenizer([n2], **self.tok_args)[0]
        n3 = self.tokenizer([n3], **self.tok_args)[0]
        # print(anchor)
        return [torch.tensor(a), 
                torch.tensor(p), 
                torch.tensor(n1),
                torch.tensor(n2),
                torch.tensor(n3)] 

    def _graphcodebert_getitem(self, a, p, n1, n2, n3):
        nl_ids = self._graphcodebert_proc_text(nl=a) # nl
        pos_code_ids, pos_attn_mask, pos_position_idx = self._graphcodebert_code_encode(code_and_dfg=p) # pos
        neg1_code_ids, neg1_attn_mask, neg1_position_idx = self._graphcodebert_code_encode(code_and_dfg=n1) 
        neg2_code_ids, neg2_attn_mask, neg2_position_idx = self._graphcodebert_code_encode(code_and_dfg=n2)
        neg3_code_ids, neg3_attn_mask, neg3_position_idx = self._graphcodebert_code_encode(code_and_dfg=n3)

        return [
                torch.tensor(pos_code_ids),
                torch.tensor(pos_attn_mask),
                torch.tensor(pos_position_idx),
                torch.tensor(neg1_code_ids),
                torch.tensor(neg1_attn_mask),
                torch.tensor(neg1_position_idx),
                torch.tensor(neg2_code_ids),
                torch.tensor(neg2_attn_mask),
                torch.tensor(neg2_position_idx),
                torch.tensor(neg3_code_ids),
                torch.tensor(neg3_attn_mask),
                torch.tensor(neg3_position_idx),
                torch.tensor(nl_ids),
               ]
    
    def __getitem__(self, item: int):
        """combined get item for all 3 models: CodeBERT, GraphCodeBERT, UniXcoder.
        if curriculum is turned off then just use hard negatives all the time."""
        anchor = self.data[item]["intent"]
        pos = self.data[item]["snippet"]
        neg1 = self._sample_soft_neg(anchor)
        neg2 = self._sample_soft_neg(anchor)
        neg3 = self._sample_soft_neg(anchor)
        anchor = self._proc_text(anchor)
        pos = self._proc_code(pos)
        neg1 = self._proc_code(neg1)
        neg2 = self._proc_code(neg2)
        neg3 = self._proc_code(neg3)
        if self.model_name == "codebert":
            return self._codebert_getitem(anchor, pos, neg1, neg2, neg3)
        elif self.model_name == "graphcodebert":
            return self._graphcodebert_getitem(anchor, pos, neg, neg2, neg3)
        elif self.model_name == "unixcoder":
            return self._unixcoder_getitem(anchor, pos, neg1, neg2, neg3)
        
# dataset class for CodeRetriever objective based training with quads and without unimodal loss.
class CodeRetrieverQuadsDataset(AllModelsDataset):
    def __init__(self, path: str, model_name: str, 
                 tokenizer=None, **tok_args):
        super(CodeRetrieverQuadsDataset, self).__init__(
            path=path, model_name=model_name,
            tokenizer=tokenizer, **tok_args,
        )
        # create a mapping of NL to all associated PLs. 
        self.intent_to_code = {}
        for rec in self.data:
            try: intent = rec["intent"]
            except TypeError: intent = rec[0]
            try: snippet = rec["snippet"]
            except TypeError: snippet = rec[1]
            try: self.intent_to_code[intent].append(snippet)
            except KeyError: self.intent_to_code[intent] = [snippet]
        
    def reset(self): pass # just for API consistency

    def _sample_soft_neg(self, intent: str):
        """sample a soft negative: first sample a random intent then sample a random negative"""
        sampled_intent = intent
        while sampled_intent == intent:
            sampled_intent = random.sample(list(self.intent_to_code.keys()), k=1)[0]
        return random.sample(self.intent_to_code[sampled_intent], k=1)[0]
                
    def _codebert_getitem(self, a, p, p_, n):
        # special tokens are added by default.
        a = self.tokenizer(a, **self.tok_args)
        p = self.tokenizer(p, **self.tok_args)
        p_ = self.tokenizer(p_, **self.tok_args)
        n = self.tokenizer(n, **self.tok_args)
        return [
            a["input_ids"][0], a["attention_mask"][0], 
            p["input_ids"][0], p["attention_mask"][0],
            p_["input_ids"][0], p_["attention_mask"][0],
            n["input_ids"][0], n["attention_mask"][0],
        ]

    def _unixcoder_getitem(self, a, p, p_, n):
        # special tokens are added by default.
        a = self.tokenizer([a], **self.tok_args)[0]
        p = self.tokenizer([p], **self.tok_args)[0]
        p_ = self.tokenizer([p_], **self.tok_args)[0]
        n = self.tokenize([n], **self.tok_args)[0]
        # print(anchor)
        return [torch.tensor(a), 
                torch.tensor(p), 
                torch.tensor(p_),
                torch.tensor(n)] 

    def _graphcodebert_getitem(self, a, p, p_, n):
        nl_ids = self._graphcodebert_proc_text(nl=a) # nl
        pos_code_ids, pos_attn_mask, pos_position_idx = self._graphcodebert_code_encode(code_and_dfg=p) # pos
        _pos_code_ids, _pos_attn_mask, _pos_position_idx = self._graphcodebert_code_encode(code_and_dfg=p_) # soft neg
        neg_code_ids, neg_attn_mask, neg_position_idx = self._graphcodebert_code_encode(code_and_dfg=n) # hard neg

        return [
                torch.tensor(pos_code_ids),
                torch.tensor(pos_attn_mask),
                torch.tensor(pos_position_idx),
                torch.tensor(_pos_code_ids),
                torch.tensor(_pos_attn_mask),
                torch.tensor(_pos_position_idx),
                torch.tensor(neg_code_ids),
                torch.tensor(neg_attn_mask),
                torch.tensor(neg_position_idx),
                torch.tensor(nl_ids),
               ]
    
    def __getitem__(self, item: int):
        """combined get item for all 3 models: CodeBERT, GraphCodeBERT, UniXcoder.
        if curriculum is turned off then just use hard negatives all the time."""
        anchor = self.data[item]["intent"]
        pos = self.data[item]["snippet"]
        neg1 = self._sample_soft_neg(anchor)
        neg2 = self._sample_soft_neg(anchor)
        anchor = self._proc_text(anchor)
        pos = self._proc_code(pos)
        neg1 = self._proc_code(neg1)
        neg2 = self._proc_code(neg2)
        if self.model_name == "codebert":
            return self._codebert_getitem(anchor, pos, neg1, neg2)
        elif self.model_name == "graphcodebert":
            return self._graphcodebert_getitem(anchor, pos, neg, neg2)
        elif self.model_name == "unixcoder":
            return self._unixcoder_getitem(anchor, pos, neg1, neg2)

# Unimodal-Bimodal and hard negatives.
class UniBiHardNegDataset(AllModelsDataset):
    """a (anchor): NL
    a+ : NL similar to a (>1, need a means to select 1) (not directly used as it is not a part of the dataset, only used to derive candidate hard negatives)
    p: PL corresponding to NL
    p+: other PLs associated with a (>1, need a means to select 1)
    n: PLs associated with a+, PLs derived by perturbations (>1, need a means to select 1)
    d_ap: pairwise distances: soft negatives bimodal
    d_an: pairwise distances: hard negatives bimodal
    d_pp+: pairwise distances: soft negatives unimodal
    d_pn:  pairwise distances: hard negatives unimodal"""
    def __init__(self, nl_code_path: str, code_syns_path: str, 
                 model_name: str, tokenizer=None, model=None, 
                 sim_intents_map: dict={}, perturbed_codes: dict={}, 
                 batch_size: int=64, device: str="cuda:0", **tok_args):
        super(UniBiHardNegDataset, self).__init__(
            path=nl_code_path, model_name=model_name,
            tokenizer=tokenizer, **tok_args,
        )
        self.pdist = nn.PairwiseDistance()
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.nl_code_path = nl_code_path
        self.code_syns_path = code_syns_path
        self.sim_intents_map = sim_intents_map
        self.perturbed_codes = perturbed_codes
        # create a mapping of NL to all associated PLs. 
        self.intent_to_code = {}
        for rec in self.data:
            try: intent = rec["intent"]
            except TypeError: intent = rec[0]
            try: snippet = rec["snippet"]
            except TypeError: snippet = rec[1]
            try: self.intent_to_code[intent].append(snippet)
            except KeyError: self.intent_to_code[intent] = [snippet]
            
        self.code_synsets = CodeSynsets(code_syns_path)
        self.train_data = copy.deepcopy(self.data)
        self.data = create_app_from_csyn_ncp(
            self.train_data, 
            self.code_synsets,
        )
        
    def _get_hard_negs(self, NL: str, PL: str) -> Tuple[List[str], List[int]]:
        rindex = 0
        code_cands: List[str] = []
        rule_cands: List[int] = []
        for tup in self.perturbed_codes.get(PL,[]): # codes from AST.
            # if self.ignore_worst_rules and tup[1] in WORST_RULES_LIST: continue
            # elif self.ignore_non_disco_rules and tup[1] in DISCO_IGNORE_LIST: continue
            rule_index = int(tup[1].replace("rule",""))
            code_cands.append(tup[0])
            rule_cands.append(rule_index)
        sim_intents: List[str] = self.sim_intents_map.get(NL,[])
        for intent, _ in sim_intents:
            code_cands += self.intent_to_code[intent]
            rule_cands += [-1]*len(self.intent_to_code[intent])
            
        return code_cands, rule_cands
    
    def _sample_soft_neg(self, intent: str):
        """sample a soft negative: first sample a random intent then sample a random negative"""
        sampled_intent = intent
        while sampled_intent == intent:
            sampled_intent = random.sample(list(self.intent_to_code.keys()), k=1)[0]
        return random.sample(self.intent_to_code[sampled_intent], k=1)[0]
    
    def __getitem__(self, item: int):
        """combined get item for all 3 models: CodeBERT, GraphCodeBERT, UniXcoder"""
        a = self._proc_text(self.data[item][0]) # a
        p = self._proc_code(self.data[item][1]) # p
        p_ = self._proc_code(self.data[item][2]) # p+
        neg_code_cands, neg_rule_cands = self._get_hard_negs(
            NL=self.data[item][0],
            PL=self.data[item][1],
        )
        if len(neg_code_cands) == 0:
            n = self._proc_code(self._sample_soft_neg(a))
            r = torch.as_tensor(0)
        elif len(neg_code_cands) == 1:
            n = self._proc_code(neg_code_cands[0])
            r = torch.as_tensor(neg_rule_cands[0])
        else:
            self.model.eval()
            with torch.no_grad():
                enc_text = torch.stack(self.model.encode_emb(
                    [a], mode="text", 
                    device_id=self.device,
                    batch_size=self.batch_size,
                )) # 1 x hidden_size
                enc_code = torch.stack(self.model.encode_emb(
                    neg_code_cands, mode="code",
                    batch_size=self.batch_size,
                    device_id=self.device,
                )) # num_cands x hidden_size
                enc_text = enc_text.repeat(len(enc_code), 1) # num_cands x hidden_size
            i = self.pdist(enc_text, enc_code).argmin().cpu().item()
            n = self._proc_code(neg_code_cands[i])
            r = torch.as_tensor(neg_rule_cands[i])
        
        if self.model_name == "codebert":
            return self._codebert_getitem(a, p, p_, n, r)
        elif self.model_name == "graphcodebert":
            return self._graphcodebert_getitem(a, p, p_, n, r)
        elif self.model_name == "unixcoder":
            return self._unixcoder_getitem(a, p, p_, n, r)
        
    def _codebert_getitem(self, a, p, p_, n, rindex: int):
        # special tokens are added by default.
        a = self.tokenizer(a, **self.tok_args)
        p = self.tokenizer(p, **self.tok_args)
        p_ = self.tokenizer(p_, **self.tok_args)
        n = self.tokenizer(n, **self.tok_args)
        return [
            a["input_ids"][0], a["attention_mask"][0], 
            p["input_ids"][0], p["attention_mask"][0],
            p_["input_ids"][0], p_["attention_mask"][0],
            n["input_ids"][0], n["attention_mask"][0], 
            rindex,
        ]

    def _unixcoder_getitem(self, a, p, p_, n, rindex: int):
        # special tokens are added by default.
        a = self.tokenizer([a], **self.tok_args)[0]
        p = self.tokenizer([p], **self.tok_args)[0]
        p_ = self.tokenizer([p_], **self.tok_args)[0]
        n = self.tokenize([n], **self.tok_args)[0]
        # print(anchor)
        return [torch.tensor(a), 
                torch.tensor(p), 
                torch.tensor(p_),
                torch.tensor(n),
                rindex] 

    def _graphcodebert_getitem(self, a, p, p_, n, rindex: int):
        nl_ids = self._graphcodebert_proc_text(nl=a) # nl
        pos_code_ids, pos_attn_mask, pos_position_idx = self._graphcodebert_code_encode(code_and_dfg=p) # pos
        _pos_code_ids, _pos_attn_mask, _pos_position_idx = self._graphcodebert_code_encode(code_and_dfg=p_) # soft neg
        neg_code_ids, neg_attn_mask, neg_position_idx = self._graphcodebert_code_encode(code_and_dfg=n) # hard neg

        return [
                torch.tensor(pos_code_ids),
                torch.tensor(pos_attn_mask),
                torch.tensor(pos_position_idx),
                torch.tensor(_pos_code_ids),
                torch.tensor(_pos_attn_mask),
                torch.tensor(_pos_position_idx),
                torch.tensor(neg_code_ids),
                torch.tensor(neg_attn_mask),
                torch.tensor(neg_position_idx),
                torch.tensor(nl_ids), rindex,
               ]
    
# dataset class for CodeRetriever objective based training.
class DiscoDataset(AllModelsDataset):
    def __init__(self, path: str, model_name: str, tokenizer=None, 
                 perturbed_codes: dict={}, **tok_args):
        # the dataset pointed to the path should contain the NL-PL pairs.
        super(DiscoDataset, self).__init__(
            path=path, model_name=model_name,
            tokenizer=tokenizer, **tok_args,
        )
        self.data_path = path
        triples = []
        self.intent_to_code = {}
        for rec in self.data:
            try: intent = rec["intent"]
            except TypeError: intent = rec[0]
            try: snippet = rec["snippet"]
            except TypeError: snippet = rec[1]
            try: self.intent_to_code[intent].append(snippet)
            except KeyError: self.intent_to_code[intent] = [snippet]
        
        for rec in self.data:
            a = rec["intent"]
            p = rec["snippet"]
            hard_negs = perturbed_codes[p]
            for n,r in hard_negs: 
                if r in DISCO_IGNORE_LIST: continue
                r = int(r.replace("rule",""))
                triples.append((a,p,n,r))
            if len(hard_negs) == 0: 
                n = self._sample_soft_neg(a)
                triples.append((a,p,n,0))
        self.data = triples
        
    def _sample_soft_neg(self, intent: str):
        """sample a soft negative: first sample a random intent then sample a random negative"""
        sampled_intent = intent
        while sampled_intent == intent:
            sampled_intent = random.sample(list(self.intent_to_code.keys()), k=1)[0]
        return random.sample(self.intent_to_code[sampled_intent], k=1)[0]
        
    def reset(self):
        """reset code pairs"""
        self.data = create_apn_from_ccp_ncp(self.train_data, self.code_pairs)
        
    def __getitem__(self, item: int):
        """combined get item for all 3 models: CodeBERT, GraphCodeBERT, UniXcoder.
        if curriculum is turned off then just use hard negatives all the time."""
        anchor = self.data[item][0]
        pos = self.data[item][1]
        neg = self.data[item][2]
        rindex = self.data[item][3]
        anchor = self._proc_text(anchor)
        pos = self._proc_code(pos)
        neg = self._proc_code(neg)
        if self.model_name == "codebert":
            return self._codebert_getitem(anchor, pos, neg, rindex)
        elif self.model_name == "graphcodebert":
            return self._graphcodebert_getitem(anchor, pos, neg, rindex)
        elif self.model_name == "unixcoder":
            return self._unixcoder_getitem(anchor, pos, neg, rindex)