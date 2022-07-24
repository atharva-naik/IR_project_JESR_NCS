#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Atharva Naik (18CS10067)

# code for creating Dataset instance for the dataloader.
import os
import torch
from typing import *
from tqdm import tqdm
from datautils.utils import *
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

# list of available models. 
MODEL_OPTIONS = ["codebert", "graphcodebert", "unixcoder"]
class MixingRate:
    """wait for `patience` steps, before returning True in __call__"""
    def __init__(self, start_ratio: int=50):
        self.start_ratio = start_ratio
        self.all_ratios = [50, 50, 25, 25, 20, 10, 10, 10, 5, 5, 4, 4, 3, 3, 2, 2, 1]
        # [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
        self.ctr = 0
        self.i = self.all_ratios.index(start_ratio)
    
    def __str__(self):
        return f"{self.ctr}/{self.all_ratios[self.i]}"
    
    def update(self):
        self.i = min(self.i+1, len(self.all_ratios)-1)
        self.ctr = 0
        
    def __call__(self):
        state = False
        if self.ctr == self.all_ratios[self.i]: 
            state = True
            self.ctr = -1
        self.ctr += 1
        
        return state
    
# update milestone.
class MilestoneUpdater:
    def __init__(self, warmup_steps: int=100):
        self.i = 0
        self.steps = [0.05+i*0.05 for i in range(20)]
        self.mixing_rate = MixingRate()
        self.warmup_steps = warmup_steps
        
    def __str__(self):
        return f"{self.mixing_rate}, next update at (acc >= {self.steps[self.i]})"
        
    def rate(self):
        return self.mixing_rate()
        
    def update(self, acc: float):
        if self.warmup_steps > 0:
            self.warmup_steps -= 1
            return
        if acc >= self.steps[self.i]:
            self.i += 1
            self.mixing_rate.update()
        
# NL-PL pairs dataset class.
class DynamicTriplesDataset(Dataset):
    def __init__(self, path: str, model_name: str, model=None,
                 sim_intents_map: Dict[str, List[str]]={}, 
                 perturbed_codes: Dict[str, List[str]]={},
                 tokenizer=None, device: str="cuda:0",
                 use_AST: bool=False, val: bool=False, 
                 **tok_args):
        super(DynamicTriplesDataset, self).__init__()
        assert model_name in MODEL_OPTIONS
        self.model_name = model_name
        self.milestone_updater = MilestoneUpdater()
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
        # (all codes associated with a given intent)
        self.intent_to_code = {}
        for rec in self.data:
            intent = rec["intent"]
            snippet = rec["snippet"]
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
        
    def update(self, acc: float):
        self.milestone_updater.update(acc)
        
    def mix_step(self):
        if self.milestone_updater.warmup_steps > 0: 
            return f"warmup({self.milestone_updater.warmup_steps}) {self.milestone_updater.mixing_rate}({self.milestone_updater.steps[self.milestone_updater.i]}) "
        else: return f"mix: {self.milestone_updater.mixing_rate}({self.milestone_updater.steps[self.milestone_updater.i]}) "
        
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
        
    def _retrieve_best_triplet(self, NL: str, PL: str, use_AST: bool, batch_size: int=48):
        sim_intents: List[str] = self.sim_intents_map[NL]
        codes_for_sim_intents: List[str] = []
        for intent, _ in sim_intents:
            codes_for_sim_intents += self.intent_to_code[intent]
        if use_AST:
            codes_for_sim_intents += self.perturbed_codes[PL] # codes from AST.
        self.model.eval()
        with torch.no_grad():
            enc_text = torch.stack(self.model.encode_emb([NL], mode="text", 
                                                         batch_size=batch_size,
                                                         device_id=self.device)) # 1 x hidden_size
            enc_codes = torch.stack(self.model.encode_emb(
                codes_for_sim_intents, mode="code", 
                device_id=self.device, batch_size=batch_size
            )) # num_cands x hidden_size
            scores = enc_text @ enc_codes.T # 1 x num_cands
        i: int = torch.topk(scores, k=1).indices[0].item()
        # print(PL)
        # print(i, codes_for_sim_intents[i])
        return NL, PL, codes_for_sim_intents[i]
    
    def _sample_rand_triplet(self, NL: str, PL: str):
        codes = []
        for intent in self.intent_to_code:
            if intent != NL: 
                codes += self.intent_to_code[intent]
                
        return NL, PL, random.choice(codes)
        
    def __getitem__(self, item: int):
        # combined get item for all 3 models: CodeBERT, GraphCodeBERT, UniXcoder.
        if self.val or self.milestone_updater.rate() == False:
            anchor, pos, neg = self._sample_rand_triplet(
                NL=self.data[item]["intent"], 
                PL=self.data[item]["snippet"],
            )
            hard_neg = False
        else:
            anchor, pos, neg = self._retrieve_best_triplet(
                NL=self.data[item]["intent"], 
                PL=self.data[item]["snippet"],
                use_AST=self.use_AST,
            )
            hard_neg = True
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
        
    def _graphcodebert_getitem(self, anchor: str, pos: Union[str, list], neg: Union[str, list]):
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