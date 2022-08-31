#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Atharva Naik (18CS10067)

# code for creating Dataset instance for the dataloader.
import os
import torch
import random
import numpy as np
from typing import *
from tqdm import tqdm
from datautils.utils import *
import torch.nn.functional as F
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader

# list of available models. 
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
    neg_3 = []
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
        self.model_name = model_name
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
        
# NL-PL pairs dataset class.
class DynamicTriplesDataset(Dataset):
    def __init__(self, path: str, model_name: str, model=None, tokenizer=None,
                 use_AST=False, val=False, warmup_steps=3000, beta=0.001, p=2,
                 sim_intents_map={}, perturbed_codes={}, device="cuda:0",
                 win_size=20, delta=0.5, epsilon=0.8, **tok_args):
        super(DynamicTriplesDataset, self).__init__()
        assert model_name in MODEL_OPTIONS
        self.model_name = model_name
        self.warmup_steps = warmup_steps
        # self.milestone_updater = MilestoneUpdater()
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
        self.soft_neg_weight = 0.8
        self.hard_neg_weight = 0.2
        self.soft_acc_buffer = np.zeros(20) # buffer of fixed size 20 
        self.hard_acc_buffer = np.zeros(20) # buffer of fixed size 20
        self.model = model # pointer to model instance to find closest NL & PL examples
        self.sim_intents_map = sim_intents_map
        self.perturbed_codes = perturbed_codes
        self.tok_args = tok_args
        self.use_AST = use_AST
        self.device = device
        self.val = val
        # if filename endswith jsonl:
        if path.endswith(".jsonl"):
            self.data = read_jsonl(path) # NL-PL pairs.
        # if filename endswith json:
        elif path.endswith(".json"):
            self.data = json.load(open(path)) # NL-PL pairs.
        # create a mapping of NL to all associated PLs. 
        self.intent_to_code = {}
        for rec in self.data:
            try: intent = rec["intent"]
            except TypeError: intent = rec[0]
            try: snippet = rec["snippet"]
            except TypeError: snippet = rec[1]
            try: self.intent_to_code[intent].append(snippet)
            except KeyError: self.intent_to_code[intent] = [snippet]
        # parser is needed for GraphCodeBERT to get the dataflow.
        if model_name == "graphcodebert":
            from datautils.parser import DFG_python
            from tree_sitter import Language, Parser
            PARSER =  Parser()
            LANGUAGE = Language('datautils/py_parser.so', 'python')
            PARSER.set_language(LANGUAGE)
            self.parser = [PARSER, DFG_python]
        if isinstance(tokenizer, RobertaTokenizer): 
            self.tokenizer = tokenizer
        elif isinstance(tokenizer, str):
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
        else: self.tokenizer = tokenizer
        
    def update(self, soft_acc: float, hard_acc: float):
        # self.milestone_updater.update(acc)
        self.soft_master_rate.push_acc(soft_acc)
        self.hard_master_rate.push_acc(hard_acc)
        if self.warmup_steps > 0: 
            self.warmup_steps -= 1
            return 
        a_s = self.soft_master_rate.attn(self.hard_master_rate)
        a_h = self.hard_master_rate.attn(self.soft_master_rate)
        attn_dist = F.softmax(torch.as_tensor([a_s, a_h]), dim=0)
        uniform = torch.as_tensor([0.8, 0.2])
        weights = (1-self.epsilon)*attn_dist + self.epsilon*uniform
        weights = weights.numpy()
        self.soft_neg_weight = weights[0]
        self.hard_neg_weight = 1-weights[0]
        
    def mix_step(self):
        # if self.milestone_updater.warmup_steps > 0: 
        #     return f"warmup({self.milestone_updater.warmup_steps}) {self.milestone_updater.mixing_rate}({self.milestone_updater.steps[self.milestone_updater.i]}) "
        # else: return f"mix: {self.milestone_updater.mixing_rate}({self.milestone_updater.steps[self.milestone_updater.i]}) "
        if self.warmup_steps > 0:
            return f"warmup({self.warmup_steps}) "
        else: return f"{self.soft_neg_weight:.3f}|{self.hard_neg_weight:.3f} "
        
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
        
    def _retrieve_best_triplet(self, NL: str, PL: str, use_AST: bool, 
                               batch_size: int=48, stochastic=True,
                               backup_neg: Union[str, None]=None):
        codes_for_sim_intents: List[str] = []
        if use_AST: # when using AST only use AST.
            codes_for_sim_intents += self.perturbed_codes[PL] # codes from AST.
            # print(PL)
            # print(codes_for_sim_intents)
        else: # TODO: add a flag for IDNS.
            sim_intents: List[str] = self.sim_intents_map[NL]
            for intent, _ in sim_intents:
                codes_for_sim_intents += self.intent_to_code[intent]
        # print("codes_for_sim_intents:", codes_for_sim_intents)
        if len(codes_for_sim_intents) == 0: # if no pool of backup candidates is available.
            neg = backup_neg
        else:
            self.model.eval()
            if len(codes_for_sim_intents) == 1: codes_for_sim_intents += backup_neg
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
                    return NL, PL, backup_neg
            else:
                i: int = torch.topk(scores, k=1).indices[0].item()
            neg = codes_for_sim_intents[i]

        return NL, PL, neg
    
    def _sample_rand_triplet(self, NL: str, PL: str):
        codes = []
        for intent in self.intent_to_code:
            if intent != NL: 
                codes += self.intent_to_code[intent]
                
        return NL, PL, random.choice(codes)
        
    def __getitem__(self, item: int):
        # combined get item for all 3 models: CodeBERT, GraphCodeBERT, UniXcoder.
        if self.val or self.warmup_steps > 0:
            hard_neg = False
        else:
            hard_neg = np.random.choice(
                [False, True], p=[
                    self.soft_neg_weight, 
                    self.hard_neg_weight,
                ])
        if hard_neg: # sample hard similar intent or AST based negatives.
            # anchor, pos, neg = self._retrieve_best_triplet(
            #     NL=self.data[item]["intent"], 
            #     PL=self.data[item]["snippet"],
            #     use_AST=self.use_AST,
            # )
            anchor, pos, neg = self._retrieve_best_triplet(
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
        anchor = self.model.embed_model.tokenize([anchor], **self.tok_args)[0]
        pos = self.model.embed_model.tokenize([pos], **self.tok_args)[0]
        neg = self.model.embed_model.tokenize([neg], **self.tok_args)[0]
        # print(anchor)
        return [torch.tensor(anchor), 
                torch.tensor(pos), 
                torch.tensor(neg),
                torch.tensor(hard_neg)] 
        
    def _graphcodebert_getitem(self, anchor: str, pos: Union[str, list], neg: Union[str, list], hard_neg: bool):
        args = self.tok_args
        tokenizer = self.tokenizer
        # nl
        nl=anchor
        nl_tokens=tokenizer.tokenize(nl)[:args["nl_length"]-2]
        nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
        nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = args["nl_length"] - len(nl_ids)
        nl_ids+=[tokenizer.pad_token_id]*padding_length 
        # pos
        code_tokens, dfg = pos
        code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
        ori2cur_pos={}
        ori2cur_pos[-1]=(0,0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
        code_tokens=[y for x in code_tokens for y in x]  
        # truncating
        code_tokens=code_tokens[:args["code_length"]+args["data_flow_length"]-2-min(len(dfg),args["data_flow_length"])]
        code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        pos_code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
        pos_position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
        dfg=dfg[:args["code_length"]+args["data_flow_length"]-len(code_tokens)]
        code_tokens+=[x[0] for x in dfg]
        pos_position_idx+=[0 for x in dfg]
        pos_code_ids+=[tokenizer.unk_token_id for x in dfg]
        padding_length=args["code_length"]+args["data_flow_length"]-len(pos_code_ids)
        pos_position_idx+=[tokenizer.pad_token_id]*padding_length
        pos_code_ids+=[tokenizer.pad_token_id]*padding_length    
        # reindex
        reverse_index={}
        for idx,x in enumerate(dfg):
            reverse_index[x[1]]=idx
        for idx,x in enumerate(dfg):
            dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
        dfg_to_dfg=[x[-1] for x in dfg]
        dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
        length=len([tokenizer.cls_token])
        dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code] 
        # calculate graph-guided masked function
        pos_attn_mask=np.zeros((args["code_length"]+args["data_flow_length"],
                                args["code_length"]+args["data_flow_length"]),
                                dtype=bool)
        # calculate begin index of node and max length of input
        node_index=sum([i>1 for i in pos_position_idx])
        max_length=sum([i!=1 for i in pos_position_idx])
        # sequence can attend to sequence
        pos_attn_mask[:node_index,:node_index]=True
        # special tokens attend to all tokens
        for idx,i in enumerate(pos_code_ids):
            if i in [0,2]:
                pos_attn_mask[idx,:max_length]=True
        # nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(dfg_to_code):
            if a<node_index and b<node_index:
                pos_attn_mask[idx+node_index,a:b]=True
                pos_attn_mask[a:b,idx+node_index]=True
        # nodes attend to adjacent nodes 
        for idx,nodes in enumerate(dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(pos_position_idx):
                    pos_attn_mask[idx+node_index,a+node_index]=True

        # neg
        code_tokens, dfg = neg
        code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
        ori2cur_pos={}
        ori2cur_pos[-1]=(0,0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
        code_tokens=[y for x in code_tokens for y in x]  
        # truncating
        code_tokens=code_tokens[:args["code_length"]+args["data_flow_length"]-2-min(len(dfg),args["data_flow_length"])]
        code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        neg_code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
        neg_position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
        dfg=dfg[:args["code_length"]+args["data_flow_length"]
                -len(code_tokens)]
        code_tokens+=[x[0] for x in dfg]
        neg_position_idx+=[0 for x in dfg]
        neg_code_ids+=[tokenizer.unk_token_id for x in dfg]
        padding_length=args["code_length"]+args["data_flow_length"]-len(neg_code_ids)
        neg_position_idx+=[tokenizer.pad_token_id]*padding_length
        neg_code_ids+=[tokenizer.pad_token_id]*padding_length    
        # reindex
        reverse_index={}
        for idx,x in enumerate(dfg):
            reverse_index[x[1]]=idx
        for idx,x in enumerate(dfg):
            dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
        dfg_to_dfg=[x[-1] for x in dfg]
        dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
        length=len([tokenizer.cls_token])
        dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code] 

        # calculate graph-guided masked function
        neg_attn_mask=np.zeros((args["code_length"]+args["data_flow_length"],
                            args["code_length"]+args["data_flow_length"]),dtype=bool)
        # calculate begin index of node and max length of input
        node_index=sum([i>1 for i in neg_position_idx])
        max_length=sum([i!=1 for i in neg_position_idx])
        # sequence can attend to sequence
        neg_attn_mask[:node_index,:node_index]=True
        # special tokens attend to all tokens
        for idx,i in enumerate(neg_code_ids):
            if i in [0,2]:
                neg_attn_mask[idx,:max_length]=True
        # nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(dfg_to_code):
            if a<node_index and b<node_index:
                neg_attn_mask[idx+node_index,a:b]=True
                neg_attn_mask[a:b,idx+node_index]=True
        # nodes attend to adjacent nodes 
        for idx,nodes in enumerate(dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(neg_position_idx):
                    neg_attn_mask[idx+node_index,a+node_index]=True

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