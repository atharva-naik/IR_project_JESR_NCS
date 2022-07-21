#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Atharva Naik - finetuning and model code.
# Soumitra Das - changes to Dataset classes for GraphCodeBERT
import os
import json
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from typing import Union, List
from tree_sitter import Language, Parser
from sklearn.metrics import ndcg_score as NDCG
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from models.metrics import recall_at_k, TripletAccuracy
from sklearn.metrics import label_ranking_average_precision_score as MRR
from datautils.parser import DFG_python
from datautils.parser import (remove_comments_and_docstrings,
                              tree_to_token_index,
                              index_to_code_token,
                              tree_to_variable_index)
from models import test_ood_performance, get_tok_path, dynamic_negative_sampling
# seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# get arguments
def get_args():
    parser = argparse.ArgumentParser("script to train (using triplet margin loss), evaluate and predict with the GraphCodeBERT in Late Fusion configuration for Neural Code Search.")    
    parser.add_argument("-en", "--exp_name", type=str, default="triplet_CodeBERT_rel_thresh", help="experiment name (will be used as folder name)")
    parser.add_argument("-c", "--candidates_path", type=str, default="candidate_snippets.json", help="path to candidates (to test retrieval)")
    parser.add_argument("-q", "--queries_path", type=str, default="query_and_candidates.json", help="path to queries (to test retrieval)")
    parser.add_argument("-tp", "--train_path", type=str, default="triples/triples_train_fixed.json", help="path to training triplet data")
    parser.add_argument("-vp", "--val_path", type=str, default="triples/triples_test_fixed.json", help="path to validation triplet data")
    parser.add_argument("-d", "--device_id", type=str, default="cpu", help="device string (GPU) for doing training/testing")
    parser.add_argument("-lr", "--lr", type=float, default=1e-5, help="learning rate for training (defaults to 1e-5)")
    parser.add_argument("-p", "--predict", action="store_true", help="flag to do prediction/testing")
    parser.add_argument("-t", "--train", action="store_true", help="flag to do training")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="no. of epochs")
    parser.add_argument("-too", "--test_ood", action="store_true", help="flat to do ood testing")
    parser.add_argument("-dns", "--dynamic_negative_sampling", action="store_true", 
                        help="do dynamic negative sampling at batch level")
    
    return parser.parse_args()
    

class GraphCodeBERTWrapperModel(nn.Module):   
    def __init__(self, encoder):
        super(GraphCodeBERTWrapperModel, self).__init__()
        self.encoder = encoder
        
    def forward(self, code_inputs=None, attn_mask=None, position_idx=None, nl_inputs=None): 
        if code_inputs is not None:
            # uses position_idx.
            nodes_mask=position_idx.eq(0)
            token_mask=position_idx.ge(2)        
            inputs_embeddings=self.encoder.embeddings.word_embeddings(code_inputs)
            nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
            nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
            avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
            inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
            return self.encoder(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx)[1]
        else:
            return self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[1]


class CodeDataset(Dataset):
    def __init__(self, code_snippets: str,  args: dict, tokenizer: Union[str, None, RobertaTokenizer]=None):
        super(CodeDataset, self).__init__()
        self.data = code_snippets
        self.args = args
        LANGUAGE = Language('datautils/py_parser.so', 'python')
        PARSER =  Parser()
        PARSER.set_language(LANGUAGE)
        self.parser = [PARSER, DFG_python]
        if isinstance(tokenizer, RobertaTokenizer):
            self.tokenizer = tokenizer
        elif isinstance(tokenizer, str):
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def proc_code(self, code: str):
        try:
            code = remove_comments_and_docstrings(code, 'python')
        except:
            print(f"error in removing comments and docstrings: {code}")
        # print(type(code))
        tree = self.parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=self.parser[1](root_node,index_to_code,{}) 
        except Exception as e:
            print("Parsing error:", e)
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG 
        
        return code_tokens, dfg
    
    def __getitem__(self, item: int):
        tokenizer = self.tokenizer
        args = self.args
        code = self.data[item]
        code_tokens,dfg=self.proc_code(code)
        code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
        ori2cur_pos={}
        ori2cur_pos[-1]=(0,0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
        code_tokens=[y for x in code_tokens for y in x]  
        #truncating
        code_tokens=code_tokens[:args["code_length"]+args["data_flow_length"]-2-min(len(dfg),args["data_flow_length"])]
        code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
        position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
        dfg=dfg[:args["code_length"]+args["data_flow_length"]
                -len(code_tokens)]
        code_tokens+=[x[0] for x in dfg]
        position_idx+=[0 for x in dfg]
        code_ids+=[tokenizer.unk_token_id for x in dfg]
        padding_length=args["code_length"]+args["data_flow_length"]-len(code_ids)
        position_idx+=[tokenizer.pad_token_id]*padding_length
        code_ids+=[tokenizer.pad_token_id]*padding_length    
        #reindex
        reverse_index={}
        for idx,x in enumerate(dfg):
            reverse_index[x[1]]=idx
        for idx,x in enumerate(dfg):
            dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
        dfg_to_dfg=[x[-1] for x in dfg]
        dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
        length=len([tokenizer.cls_token])
        dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code] 

        #calculate graph-guided masked function
        attn_mask=np.zeros((self.args["code_length"]+self.args["data_flow_length"],
                            self.args["code_length"]+self.args["data_flow_length"]),dtype=bool)
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in position_idx])
        max_length=sum([i!=1 for i in position_idx])
        #sequence can attend to sequence
        attn_mask[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(code_ids):
            if i in [0,2]:
                attn_mask[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(dfg_to_code):
            if a<node_index and b<node_index:
                attn_mask[idx+node_index,a:b]=True
                attn_mask[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(position_idx):
                    attn_mask[idx+node_index,a+node_index]=True  
                    
        return (torch.tensor(code_ids),
                torch.tensor(attn_mask),
                torch.tensor(position_idx))    
    
        
class TextDataset(Dataset):
    def __init__(self, texts: str, tokenizer: Union[str, None, RobertaTokenizer]=None, **tok_args):
        super(TextDataset, self).__init__()
        self.data = texts
        self.tok_args = tok_args
        if isinstance(tokenizer, RobertaTokenizer):
            self.tokenizer = tokenizer
        elif isinstance(tokenizer, str):
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def proc_text(self, text: str):
        text = " ".join(text.split("\n"))
        text = " ".join(text.split()).strip()
        return text
    
    def __getitem__(self, i: int):
        text = self.proc_text(self.data[i])
        if self.tokenizer:
            # special tokens are added by default.
            text = self.tokenizer(text, **self.tok_args)            
            return [text["input_ids"][0]]
        else:
            return [text]
        
        
class TextCodePairDataset(Dataset):
    def __init__(self, texts: str, codes: str, args: dict, tokenizer: Union[str, None, RobertaTokenizer]=None):
        super(TextCodePairDataset, self).__init__()
        self.data = [(text, code) for text, code in zip(texts, codes)]
        self.args = args
        LANGUAGE = Language('datautils/py_parser.so', 'python')
        PARSER =  Parser()
        PARSER.set_language(LANGUAGE)
        self.parser = [PARSER, DFG_python]
        if isinstance(tokenizer, RobertaTokenizer):
            self.tokenizer = tokenizer
        elif isinstance(tokenizer, str):
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def proc_code(self, code: str):
        try:
            code = remove_comments_and_docstrings(code, 'python')
        except:
            pass
        # print(type(code))
        tree = self.parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=self.parser[1](root_node,index_to_code,{}) 
        except Exception as e:
            print("Ln 246:", e)
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG 
        return code_tokens,dfg
    
    def proc_text(self, text: str):
        text = " ".join(text.split("\n"))
        text = " ".join(text.split()).strip()
        return text
    
    def __getitem__(self, item: int):
        tokenizer = self.tokenizer
        args = self.args
        text = self.data[item][0]
        code = self.data[item][1]

        code_tokens,dfg=self.proc_code(code)
        code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
        ori2cur_pos={}
        ori2cur_pos[-1]=(0,0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
        code_tokens=[y for x in code_tokens for y in x]  
        #truncating
        code_tokens=code_tokens[:args["code_length"]+args["data_flow_length"]-2-min(len(dfg),args["data_flow_length"])]
        code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
        position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
        dfg=dfg[:args["code_length"]+args["data_flow_length"]
                -len(code_tokens)]
        code_tokens+=[x[0] for x in dfg]
        position_idx+=[0 for x in dfg]
        code_ids+=[tokenizer.unk_token_id for x in dfg]
        padding_length=args["code_length"]+args["data_flow_length"]-len(code_ids)
        position_idx+=[tokenizer.pad_token_id]*padding_length
        code_ids+=[tokenizer.pad_token_id]*padding_length    
        #reindex
        reverse_index={}
        for idx,x in enumerate(dfg):
            reverse_index[x[1]]=idx
        for idx,x in enumerate(dfg):
            dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
        dfg_to_dfg=[x[-1] for x in dfg]
        dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
        length=len([tokenizer.cls_token])
        dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]  
        #nl
        nl=self.proc_text(text)
        nl_tokens=tokenizer.tokenize(nl)[:args["nl_length"]-2]
        nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
        nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = args["nl_length"] - len(nl_ids)
        nl_ids+=[tokenizer.pad_token_id]*padding_length

        #calculate graph-guided masked function
        attn_mask=np.zeros((self.args["code_length"]+self.args["data_flow_length"],
                            self.args["code_length"]+self.args["data_flow_length"]),dtype=bool)
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in position_idx])
        max_length=sum([i!=1 for i in position_idx])
        #sequence can attend to sequence
        attn_mask[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(code_ids):
            if i in [0,2]:
                attn_mask[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(dfg_to_code):
            if a<node_index and b<node_index:
                attn_mask[idx+node_index,a:b]=True
                attn_mask[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(position_idx):
                    attn_mask[idx+node_index,a+node_index]=True 

        return (torch.tensor(code_ids),
                torch.tensor(attn_mask),
                torch.tensor(position_idx),
                torch.tensor(nl_ids))
        
        
class TriplesDataset(Dataset):
    def __init__(self, path: str, args: dict, 
                 tokenizer: Union[str, None, RobertaTokenizer]=None):
        super(TriplesDataset, self).__init__()
        self.data = json.load(open(path))
        self.args = args
        LANGUAGE = Language('datautils/py_parser.so', 'python')
        PARSER =  Parser()
        PARSER.set_language(LANGUAGE)
        self.parser = [PARSER, DFG_python]
        if isinstance(tokenizer, RobertaTokenizer):
            self.tokenizer = tokenizer
        elif isinstance(tokenizer, str):
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def proc_text(self, text: str):
        text = " ".join(text.split("\n"))
        text = " ".join(text.split()).strip()
        return text
    
    def proc_code(self, code: str):
        try:
            code = remove_comments_and_docstrings(code, 'python')
        except: pass
        tree = self.parser[0].parse(bytes(code, 'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_ = self.parser[1](root_node,index_to_code,{}) 
        except Exception as e:
            print("Ln 380:", e)
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG 
        
        return code_tokens, dfg
        
    def __getitem__(self, item: int):
        tokenizer = self.tokenizer
        args = self.args
        text = self.data[item][0]
        pos = self.data[item][1]
        neg = self.data[item][2]
        # nl
        nl=self.proc_text(text)
        nl_tokens=tokenizer.tokenize(nl)[:args["nl_length"]-2]
        nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
        nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = args["nl_length"] - len(nl_ids)
        nl_ids+=[tokenizer.pad_token_id]*padding_length 
        # pos
        code_tokens,dfg=self.proc_code(pos)
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
        dfg=dfg[:args["code_length"]+args["data_flow_length"]
                -len(code_tokens)]
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
        pos_attn_mask=np.zeros((self.args["code_length"]+self.args["data_flow_length"],
                            self.args["code_length"]+self.args["data_flow_length"]),dtype=bool)
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
        code_tokens,dfg=self.proc_code(neg)
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
        neg_attn_mask=np.zeros((self.args["code_length"]+self.args["data_flow_length"],
                            self.args["code_length"]+self.args["data_flow_length"]),dtype=bool)
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
                torch.tensor(nl_ids)
               )

    
class GraphCodeBERTripletNet(nn.Module):
    """ Class to 
    1) finetune GraphCodeBERT in a late fusion setting using triplet margin loss.
    2) Evaluate metrics on unseen test set.
    3) 
    """
    def __init__(self, model_path: str="microsoft/graphcodebert-base", 
                 tok_path: str="microsoft/graphcodebert-base", **args):
        super(GraphCodeBERTripletNet, self).__init__()
        self.config = {}
        self.config["model_path"] = model_path
        self.config["tok_path"] = tok_path
        
        print(f"loading pretrained GraphCodeBERT embedding model from {model_path}")
        start = time.time()
        self.embed_model = GraphCodeBERTWrapperModel(
            RobertaModel.from_pretrained(model_path)
        )
        print(f"loaded embedding model in {(time.time()-start):.2f}s")
        print(f"loaded tokenizer files from {tok_path}")
        # create tokenizer.
        self.tokenizer = RobertaTokenizer.from_pretrained(tok_path)
        # optimizer and loss.
        adam_eps = 1e-8
        lr = args.get("lr", 1e-5)
        margin = args.get("margin", 1)
        dist_fn_deg = args.get("dist_fn_deg", 2)
        # print optimizer and loss function.
        print(f"optimizer = AdamW(lr={lr}, eps={adam_eps})")
        print(f"loss_fn = TripletMarginLoss(margin={margin}, p={dist_fn_deg})")
        # create optimizer object and loss function.
        self.optimizer = AdamW(
            self.parameters(), 
            eps=adam_eps, lr=lr
        )
        self.loss_fn = nn.TripletMarginLoss(
            margin=margin, 
            p=dist_fn_deg
        )
        # store config info.
        self.config["dist_fn_deg"] = dist_fn_deg
        self.config["optimizer"] = f"{self.optimizer}"
        self.config["loss_fn"] = f"{self.loss_fn}"
        self.config["margin"] = margin
        self.config["lr"] = lr
        
    def forward(self, anchor_title, pos_snippet, neg_snippet):
        anchor_text_emb = self.embed_model(nl_inputs=anchor_title)
        anchor_text_emb = self.embed_model(nl_inputs=anchor_title)
        x = pos_snippet
        pos_code_emb = self.embed_model(code_inputs=x[0], attn_mask=x[1], position_idx=x[2])
        x = neg_snippet
        neg_code_emb = self.embed_model(code_inputs=x[0], attn_mask=x[1], position_idx=x[2])
        
        return anchor_text_emb, pos_code_emb, neg_code_emb
        
    def val(self, valloader: DataLoader, epoch_i: int=0, epochs: int=0, device="cuda:0"):
        self.eval()
        val_acc = TripletAccuracy()
        batch_losses = []
        pbar = tqdm(enumerate(valloader), total=len(valloader), 
                    desc=f"val: epoch: {epoch_i+1}/{epochs} batch_loss: 0 loss: 0 acc: 0")
        for step, batch in pbar:
            with torch.no_grad():
                anchor_title = batch[-1].to(device)
                pos_snippet = (batch[0].to(device), batch[1].to(device), batch[2].to(device))
                neg_snippet = (batch[3].to(device), batch[4].to(device), batch[5].to(device))
                anchor_text_emb, pos_code_emb, neg_code_emb = self(anchor_title, pos_snippet, neg_snippet)
                batch_loss = self.loss_fn(anchor_text_emb, pos_code_emb, neg_code_emb)
                val_acc.update(anchor_text_emb, pos_code_emb, neg_code_emb)
                batch_losses.append(batch_loss.item())
                pbar.set_description(f"val: epoch: {epoch_i+1}/{epochs} batch_loss: {batch_loss:.3f} loss: {np.mean(batch_losses):.3f} acc: {100*val_acc.get():.2f}")
                # if step == 5: break # DEBUG
        return val_acc.get(), np.mean(batch_losses)
        
    def encode_emb(self, text_or_snippets: List[str], mode: str="text", **args):
        """Note: our late fusion GraphCodeBERT is a universal encoder for text and code, so the same function works for both."""
        batch_size = args.get("batch_size", 32)
        device_id = args.get("device_id", "cuda:0")
        device = torch.device(device_id if torch.cuda.is_available() else "cpu")
        use_tqdm = args.get("use_tqdm", False)
        self.to(device)
        self.eval()
        
        if mode == "text":
            dataset = TextDataset(text_or_snippets, tokenizer=self.tokenizer,
                                  truncation=True, padding="max_length",
                                  max_length=100, add_special_tokens=True,
                                  return_tensors="pt")

        elif mode == "code":
            dataset = CodeDataset(text_or_snippets, 
                                  tokenizer=self.tokenizer,
                                  args={
                                          "nl_length": 100, 
                                          "code_length": 100, 
                                          "data_flow_length": 64
                                       }
                                 )
        else: raise TypeError("Unrecognized encoding mode")
        
        datalloader = DataLoader(dataset, shuffle=False, 
                                 batch_size=batch_size)
        pbar = tqdm(enumerate(datalloader), total=len(datalloader), 
                    desc=f"encoding {mode}", disable=not(use_tqdm))
        all_embeds = []
        for step, batch in pbar:
            with torch.no_grad():
                if mode == "text":
                    nl_inputs = batch[0].to(device)
                    batch_embed = self.embed_model(nl_inputs=nl_inputs)
                elif mode == "code":
                    code_inputs = batch[0].to(device)
                    attn_masks = batch[1].to(device)
                    position_idx = batch[2].to(device)
                    batch_embed = self.embed_model(code_inputs=code_inputs, 
                                                   attn_mask=attn_masks, 
                                                   position_idx=position_idx)
                for embed in batch_embed: 
                    all_embeds.append(embed)
                # if step == 5: break # DEBUG
        # print(type(all_embeds[0]), len(all_embeds))
        print(len(all_embeds))
        return all_embeds
#     def joint_classify(self, text_snippets: List[str], 
#                        code_snippets: List[str], **args):
#         """The usual joint encoding setup of CodeBERT (similar to NLI)"""
#         batch_size = args.get("batch_size", 48)
#         device_id = args.get("device_id", "cuda:0")
#         device = torch.device(device_id)
#         use_tqdm = args.get("use_tqdm", False)
#         self.to(device)
#         self.eval()
        
#         dataset = TextCodePairDataset(text_snippets, code_snippets, 
#                                       tokenizer=self.tokenizer, truncation=True, 
#                                       padding="max_length", max_length=100, 
#                                       add_special_tokens=True, return_tensors="pt")
#         datalloader = DataLoader(dataset, shuffle=False, 
#                                  batch_size=batch_size)
#         pbar = tqdm(enumerate(datalloader), total=len(datalloader), 
#                     desc=f"enocding {mode}", disable=not(use_tqdm))
#         all_embeds = []
#         for step, batch in pbar:
#             with torch.no_grad():
#                 enc_args = (batch[0].to(device), batch[1].to(device))
#                 batch_embed = self.embed_model(*enc_args).pooler_output
#                 for embed in batch_embed: all_embeds.append(embed)
#                 # if step == 5: break # DEBUG
#         # print(type(all_embeds[0]), len(all_embeds))
#         return all_embeds
    def fit(self, train_path: str, val_path: str, **args):
        batch_size = args.get("batch_size", 32)
        self.config["batch_size"] = batch_size
        epochs = args.get("epochs", 5)
        self.config["epochs"] = epochs
        device_id = args.get("device_id", "cuda:0")
        self.config["device_id"] = device_id
        device = torch.device(device_id)
        exp_name = args.get("exp_name", "experiment")
        self.config["exp_name"] = exp_name
        os.makedirs(exp_name, exist_ok=True)
        save_path = os.path.join(exp_name, "model.pt")
        self.config["train_path"] = train_path
        self.config["val_path"] = val_path
        
        config_path = os.path.join(exp_name, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f)
        print(f"saved config to {config_path}")
        print(f"model will be saved at {save_path}")
        print(f"moving model to {device}")
        self.embed_model.to(device)
        trainset = TriplesDataset(train_path, tokenizer=self.tokenizer,
                                  args={
                                          "nl_length": 100, 
                                          "code_length": 100, 
                                          "data_flow_length": 64
                                 })
        valset = TriplesDataset(val_path, tokenizer=self.tokenizer,
                                args={
                                       "nl_length": 100, 
                                       "code_length": 100, 
                                       "data_flow_length": 64
                               })
        trainloader = DataLoader(trainset, shuffle=True, 
                                 batch_size=batch_size)
        valloader = DataLoader(valset, shuffle=False,
                               batch_size=batch_size)
        train_metrics = {
            "epochs": [],
            "summary": [],
        } 
        train_acc = TripletAccuracy()
        best_val_acc = 0
        for epoch_i in range(epochs):
            self.train()
            batch_losses = []
            pbar = tqdm(enumerate(trainloader), total=len(trainloader),
                        desc=f"train: epoch: {epoch_i+1}/{epochs} batch_loss: 0 loss: 0 acc: 0")
            train_acc.reset()
            for step, batch in pbar:
                if args.get("dynamic_negative_sampling", False):
                    batch = dynamic_negative_sampling(
                        self.embed_model, batch, 
                        model_name="graphcodebert", 
                        device=device, k=1
                    )
                anchor_title = batch[-1].to(device)
                pos_snippet = (batch[0].to(device), batch[1].to(device), batch[2].to(device))
                neg_snippet = (batch[3].to(device), batch[4].to(device), batch[5].to(device))
                # print(neg_snippet[0].shape, neg_snippet[1].shape, neg_snippet[2].shape)
                anchor_text_emb, pos_code_emb, neg_code_emb = self(anchor_title, pos_snippet, neg_snippet)
                batch_loss = self.loss_fn(anchor_text_emb, pos_code_emb, neg_code_emb)
                batch_loss.backward()
                self.optimizer.step()
                train_acc.update(anchor_text_emb, pos_code_emb, neg_code_emb)
                # scheduler.step()  # Update learning rate schedule
                self.zero_grad()
                batch_losses.append(batch_loss.item())
                pbar.set_description(f"{exp_name}: train: epoch: {epoch_i+1}/{epochs} batch_loss: {batch_loss:.3f} loss: {np.mean(batch_losses):.3f} acc: {100*train_acc.get():.2f}")
                # if step == 5: break # DEBUG
            # validate current model
            val_acc, val_loss = self.val(valloader, epoch_i=epoch_i, 
                                         epochs=epochs, device=device)
            if val_acc > best_val_acc:
                print(f"saving best model till now with val_acc: {val_acc} at {save_path}")
                best_val_acc = val_acc
                torch.save(self.state_dict(), save_path)
            train_metrics["epochs"].append({
                "train_batch_losses": batch_losses, 
                "train_loss": np.mean(batch_losses), 
                "train_acc": 100*train_acc.get(),
                "val_loss": val_loss,
                "val_acc": 100*val_acc,
            })
        
        return train_metrics

    
def main(args):    
    print("initializing model and tokenizer ..")
    tok_path = os.path.join(os.path.expanduser("~"), "graphcodebert-base-tok")
    print("creating model object")
    triplet_net = GraphCodeBERTripletNet(tok_path=tok_path)
    if args.train:
        print("commencing training")
        metrics = triplet_net.fit(**vars(args))
        metrics_path = os.path.join(args.exp_name, "train_metrics.json")
        print(f"saving metrics to {metrics_path}")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
    if args.predict:
        model_path = os.path.join(args.exp_name, "model.pt")
        print(model_path)
        
def test_retreival(args):
    print("initializing model and tokenizer ..")
    tok_path = os.path.join(os.path.expanduser("~"), "graphcodebert-base-tok")
    device = args.device_id if torch.cuda.is_available() else "cpu"
    
    ckpt_path = os.path.join(args.exp_name, "model.pt")
    print(f"loading checkpoint (state dict) from {ckpt_path}")
    try: state_dict = torch.load(ckpt_path, map_location="cpu")
    except Exception as e: 
        state_dict = None
        print("\x1b[31;1mCouldn't load state dict because\x1b[0m")
        print(e)
    
    print("creating model object")
    triplet_net = GraphCodeBERTripletNet(tok_path=tok_path)
    if state_dict: 
        print(f"\x1b[32;1msuccesfully loaded state dict from {ckpt_path}\x1b[0m")
        triplet_net.load_state_dict(state_dict)
    print(f"loading candidates from {args.candidates_path}")
    code_and_annotations = json.load(open(args.candidates_path))
    
    for setting in ["code", "annot", "code+annot"]:
        if setting == "code":
            candidates = code_and_annotations["snippets"]
        elif setting == "annot":
            candidates = code_and_annotations["annotations"]
        else: # use both code and annotations.
            code_candidates = code_and_annotations["snippets"]
            annot_candidates = code_and_annotations["annotations"]
            candidates = code_candidates

        print(f"loading queries from {args.queries_path}")
        queries_and_cand_labels = json.load(open(args.queries_path))
        queries = [i["query"] for i in queries_and_cand_labels]
        labels = [i["docs"] for i in queries_and_cand_labels]
        # dist_func = "l2_dist"
        for dist_func in ["l2_dist", "inner_prod"]:
            metrics_path = os.path.join(args.exp_name, f"test_metrics_{dist_func}_{setting}.json")
            # if dist_func in ["l2_dist", "inner_prod"]:
            print(f"encoding {len(queries)} queries:")
            query_mat = triplet_net.encode_emb(queries, mode="text", 
                                               use_tqdm=True, **vars(args))
            query_mat = torch.stack(query_mat)

            print(f"encoding {len(candidates)} candidates:")
            if setting == "code":
                cand_mat = triplet_net.encode_emb(candidates, mode="code", 
                                                  use_tqdm=True, **vars(args))
                cand_mat = torch.stack(cand_mat)
            elif setting == "annot":
                cand_mat = triplet_net.encode_emb(candidates, mode="text", 
                                                  use_tqdm=True, **vars(args))
                cand_mat = torch.stack(cand_mat)
            else:
                cand_mat_code = triplet_net.encode_emb(code_candidates, mode="code", 
                                                       use_tqdm=True, **vars(args))
                cand_mat_annot = triplet_net.encode_emb(annot_candidates, mode="text", 
                                                        use_tqdm=True, **vars(args))
                cand_mat_code = torch.stack(cand_mat_code)
                cand_mat_annot = torch.stack(cand_mat_annot)
                    # cand_mat = (cand_mat_code + cand_mat_annot)/2
            # print(query_mat.shape, cand_mat.shape)
            if dist_func == "inner_prod": 
                if setting == "code+annot":
                    scores_code = query_mat @ cand_mat_code.T
                    scores_annot = query_mat @ cand_mat_annot.T
                    scores = scores_code + scores_annot
                else:
                    scores = query_mat @ cand_mat.T
                # print(scores.shape)
            elif dist_func == "l2_dist": 
                if setting == "code+annot":
                    scores_code = torch.cdist(query_mat, cand_mat_code, p=2)
                    scores_annot = torch.cdist(query_mat, cand_mat_annot, p=2)
                    scores = scores_code + scores_annot
                else:
                    scores = torch.cdist(query_mat, cand_mat, p=2)
            # elif mode == "joint_cls": scores = triplet_net.joint_classify(queries, candidates)
            doc_ranks = scores.argsort(axis=1)
            if dist_func == "inner_prod":
                doc_ranks = doc_ranks.flip(dims=[1])
            label_ranks = []
            avg_rank = 0
            avg_best_rank = 0 
            N = 0
            M = 0

            lrap_GT = np.zeros(
                (
                    len(queries), 
                    len(candidates)
                )
            )
            recall_at_ = []
            for i in range(1,10+1):
                recall_at_.append(
                    recall_at_k(
                        labels, 
                        doc_ranks.tolist(), 
                        k=5*i
                    )
                )
            for i in range(len(labels)):
                for j in labels[i]:
                    lrap_GT[i][j] = 1

            for i, rank_list in enumerate(doc_ranks):
                rank_list = rank_list.tolist()
                # if dist_func == "inner_prod": rank_list = rank_list.tolist()[::-1]
                # elif dist_func == "l2_dist": rank_list = rank_list.tolist()
                instance_label_ranks = []
                ranks = []
                for cand_rank in labels[i]:
                    # print(rank_list, cand_rank)
                    rank = rank_list.index(cand_rank)
                    avg_rank += rank
                    ranks.append(rank)
                    N += 1
                    instance_label_ranks.append(rank)
                M += 1
                avg_best_rank += min(ranks)
                label_ranks.append(instance_label_ranks)
            metrics = {
                "avg_candidate_rank": avg_rank/N,
                "avg_best_candidate_rank": avg_best_rank/M,
                "recall": {
                    f"@{5*i}": recall_at_[i-1] for i in range(1,10+1) 
                },
            }
            print("avg canditate rank:", avg_rank/N)
            print("avg best candidate rank:", avg_best_rank/M)
            for i in range(1,10+1):
                print(f"recall@{5*i} = {recall_at_[i-1]}")
            if dist_func == "inner_prod":
                # -scores for distance based scores, no - for innert product based scores.
                mrr = MRR(lrap_GT, scores.cpu().numpy())
                ndcg = NDCG(lrap_GT, scores.cpu().numpy())
            elif dist_func == "l2_dist":
                # -scores for distance based scores, no - for innert product based scores.
                mrr = MRR(lrap_GT, -scores.cpu().numpy())
                ndcg = NDCG(lrap_GT, -scores.cpu().numpy())
                
            metrics["mrr"] = mrr
            metrics["ndcg"] = ndcg
            print("NDCG:", ndcg)
            print("MRR (LRAP):", mrr)
            if not os.path.exists(args.exp_name):
                print("missing experiment folder: assuming zero-shot setting")
                metrics_path = os.path.join(
                    "GraphCodeBERT_zero_shot", 
                    f"test_metrics_{dist_func}_{setting}.json"
                )
                os.makedirs("GraphCodeBERT_zero_shot", exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f)
#     with open("pred_cand_ranks.json", "w") as f:
#         json.dump(label_ranks, f, indent=4)
if __name__ == "__main__":
    args = get_args()
    if args.train:
        main(args=args) 
    elif args.predict:
        test_retreival(args=args)
    if args.test_ood: 
        print("creating model object")
        # instantiate model class.
        tok_path = get_tok_path("graphcodebert")
        triplet_net = GraphCodeBERTripletNet(tok_path=tok_path, **vars(args))
        test_ood_performance(
            triplet_net, model_name="graphcodebert", args=args,
            query_paths=["query_and_candidates.json", "external_knowledge/queries.json", 
                         "data/queries_webquery.json", "data/queries_codesearchnet.json"],
            cand_paths=["candidate_snippets.json", "external_knowledge/candidates.json", 
                        "data/candidates_webquery.json", "data/candidates_codesearchnet.json"], 
        )