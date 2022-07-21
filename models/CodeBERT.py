#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Atharva Naik (18CS10067)
import os
import json
import time
import torch
import models
import random
import pathlib
import argparse
import numpy as np
import transformers
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from typing import Union, List
from datautils import read_jsonl
from sklearn.metrics import ndcg_score as NDCG
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from models.metrics import TripletAccuracy, recall_at_k
from sklearn.metrics import label_ranking_average_precision_score as MRR
from models import test_ood_performance, get_tok_path, dynamic_negative_sampling

# set logging level of transformers.
transformers.logging.set_verbosity_error()
# seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# get arguments
def get_args():
    parser = argparse.ArgumentParser("script to train (using triplet margin loss), evaluate and predict with the CodeBERT in Late Fusion configuration for Neural Code Search.")
    parser.add_argument("-pp", "--predict_path", type=str, default="triples/triples_train.json", help="path to data for prediction of regression scores")
    parser.add_argument("-tp", "--train_path", type=str, default="triples/triples_train.json", help="path to training triplet data")
    parser.add_argument("-vp", "--val_path", type=str, default="triples/triples_test.json", help="path to validation triplet data")
    parser.add_argument("-c", "--candidates_path", type=str, default="candidate_snippets.json", help="path to candidates (to test retrieval)")
    parser.add_argument("-q", "--queries_path", type=str, default="query_and_candidates.json", help="path to queries (to test retrieval)")
    parser.add_argument("-en", "--exp_name", type=str, default="triplet_CodeBERT_rel_thresh", help="experiment name (will be used as folder name)")
    parser.add_argument("-d", "--device_id", type=str, default="cpu", help="device string (GPU) for doing training/testing")
    parser.add_argument("-tec", "--test_cls", action="store_true", help="")
    parser.add_argument("-tc", "--train_cls", action="store_true", help="")
    parser.add_argument("-ter", "--test_rel", action="store_true", help="")
    parser.add_argument("-tr", "--train_rel", action="store_true", help="")
    parser.add_argument("-lr", "--lr", type=float, default=1e-5, help="learning rate for training (defaults to 1e-5)")
    parser.add_argument("-te", "--test", action="store_true", help="flag to do testing")
    parser.add_argument("-t", "--train", action="store_true", help="flag to do training")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="no. of epochs")
    parser.add_argument("-dns", "--dynamic_negative_sampling", action="store_true", 
                        help="do dynamic negative sampling at batch level")
    parser.add_argument("-too", "--test_ood", action="store_true", help="flat to do ood testing")
    # parser.add_argument("-cp", "--ckpt_path", type=str, default="triplet_CodeBERT_rel_thresh/model.pt")
    return parser.parse_args()

# TripletMarginWithDistanceLoss for custom design function.
class CodeDataset(Dataset):
    def __init__(self, code_snippets: str, tokenizer: Union[str, None, RobertaTokenizer]=None, **tok_args):
        super(CodeDataset, self).__init__()
        self.data = code_snippets
        self.tok_args = tok_args
        if isinstance(tokenizer, RobertaTokenizer):
            self.tokenizer = tokenizer
        elif isinstance(tokenizer, str):
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def proc_code(self, code: str):
        code = " ".join(code.split("\n")).strip()
        return code
    
    def __getitem__(self, i: int):
        code = self.proc_code(self.data[i])
        if self.tokenizer:
            # special tokens are added by default.
            code = self.tokenizer(code, **self.tok_args)            
            return [code["input_ids"][0], 
                    code["attention_mask"][0]]
        else:
            return [code]
        
        
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
            return [text["input_ids"][0], 
                    text["attention_mask"][0]]
        else:
            return [text]
        
        
class RelevanceClassifierDataset(Dataset):
    def __init__(self, path: str, thresh: float=0.06441,
                 tokenizer: Union[str, None, RobertaTokenizer]=None, 
                 **tok_args):
        super(RelevanceClassifierDataset, self).__init__()
        self.data = json.load(open(path))
        self.tok_args = tok_args
        if isinstance(tokenizer, RobertaTokenizer):
            self.tokenizer = tokenizer
        elif isinstance(tokenizer, str):
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        self.thresh = thresh
    
    def __len__(self):
        return len(self.data)
    
    def proc_code(self, code: str):
        code = " ".join(code.split("\n")).strip()
        return code
    
    def proc_text(self, text: str):
        text = " ".join(text.split("\n"))
        text = " ".join(text.split()).strip()
        return text
    
    def __getitem__(self, i: int):
        try:
            text = self.proc_text(self.data[i]["intent"])
            code = self.proc_code(self.data[i]["snippet"])
            label = 1 if self.data[i]["prob"] >= self.thresh else 0
        except TypeError:
            print(self.data[i])
        if self.tokenizer:
            # special tokens are added by default.
            text_n_code = self.tokenizer(text, code, **self.tok_args)
            return [text_n_code["input_ids"][0], 
                    text_n_code["attention_mask"][0],
                    torch.as_tensor(label)]
        else:
            return [text_n_code, label]
        

class RelevanceRegressionDataset(Dataset):
    def __init__(self, path: str, 
                 tokenizer: Union[str, None, RobertaTokenizer]=None, 
                 **tok_args):
        super(RelevanceRegressionDataset, self).__init__()
        self.data = json.load(open(path))
        self.tok_args = tok_args
        if isinstance(tokenizer, RobertaTokenizer):
            self.tokenizer = tokenizer
        elif isinstance(tokenizer, str):
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def proc_code(self, code: str):
        code = " ".join(code.split("\n")).strip()
        return code
    
    def proc_text(self, text: str):
        text = " ".join(text.split("\n"))
        text = " ".join(text.split()).strip()
        return text
    
    def __getitem__(self, i: int):
        try:
            text = self.proc_text(self.data[i]["intent"])
            code = self.proc_code(self.data[i]["snippet"])
            label = float(self.data[i]["prob"])
        except TypeError:
            print(self.data[i])
        if self.tokenizer:
            # special tokens are added by default.
            text_n_code = self.tokenizer(text, code, **self.tok_args)
            return [text_n_code["input_ids"][0], 
                    text_n_code["attention_mask"][0],
                    torch.as_tensor(label)]
        else:
            return [text_n_code, label]
        
        
class TextCodePairDataset(Dataset):
    def __init__(self, query_candidate_pairs: str, 
                 tokenizer: Union[str, None, RobertaTokenizer]=None, 
                 **tok_args):
        super(TextCodePairDataset, self).__init__()
        self.data = query_candidate_pairs
        self.tok_args = tok_args
        if isinstance(tokenizer, RobertaTokenizer):
            self.tokenizer = tokenizer
        elif isinstance(tokenizer, str):
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def proc_code(self, code: str):
        code = " ".join(code.split("\n")).strip()
        return code
    
    def proc_text(self, text: str):
        text = " ".join(text.split("\n"))
        text = " ".join(text.split()).strip()
        return text
    
    def __getitem__(self, i: int):
        text = self.proc_text(self.data[i][0])
        code = self.proc_code(self.data[i][1])
        if self.tokenizer:
            # special tokens are added by default.
            text_n_code = self.tokenizer(text, code, **self.tok_args)
            return [text_n_code["input_ids"][0], 
                    text_n_code["attention_mask"][0]]
        else:
            return [text_n_code]
    
    
class TriplesDataset(Dataset):
    def __init__(self, path: str, tokenizer: Union[str, None, RobertaTokenizer]=None, **tok_args):
        super(TriplesDataset, self).__init__()
        self.data = json.load(open(path))
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
    
    def proc_code(self, code: str):
        code = " ".join(code.split("\n")).strip()
        return code
        
    def __getitem__(self, i: int):
        anchor = self.proc_text(self.data[i][0])
        pos = self.proc_code(self.data[i][1])
        neg = self.proc_code(self.data[i][2])
        if self.tokenizer:
            # special tokens are added by default.
            anchor = self.tokenizer(anchor, **self.tok_args)
            pos = self.tokenizer(pos, **self.tok_args)
            neg = self.tokenizer(neg, **self.tok_args)
            
            return [
                    anchor["input_ids"][0], anchor["attention_mask"][0], 
                    pos["input_ids"][0], pos["attention_mask"][0],
                    neg["input_ids"][0], neg["attention_mask"][0],
                   ]
        else:
            return [anchor, pos, neg]      
        
        
class CodeBERTRelevanceClassifier(nn.Module):
    """
    finetune CodeBERT over CoNaLa mined pairs 
    for predicting relevance score using regression
    """
    def __init__(self, model_path: str="microsoft/codebert-base", 
                 tok_path: str="microsoft/codebert-base", **args):
        super(CodeBERTRelevanceClassifier, self).__init__()
        self.config = {}
        self.config["tok_path"] = tok_path
        self.config["model_path"] = model_path
        
        print(f"loading pretrained CodeBERT embedding model from {model_path}")
        start = time.time()
        self.model = RobertaModel.from_pretrained(model_path)
        print(f"loaded CodeBERT model in {(time.time()-start):.2f}s")
        print(f"loaded tokenizer files from {tok_path}")
        self.mlp = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
        self.tokenizer = RobertaTokenizer.from_pretrained(tok_path)
        # optimizer and loss.
        adam_eps = 1e-8
        lr = args.get("lr", 1e-5)
        self.config["lr"] = lr
        
        print(f"optimizer = AdamW(lr={lr}, eps={adam_eps})")
        self.optimizer = AdamW(self.parameters(), eps=adam_eps, lr=lr)
        self.config["optimizer"] = f"{self.optimizer}"
        
        self.loss_fn = nn.BCELoss()
        print(f"loss_fn = {self.loss_fn}")
        self.config["loss_fn"] = f"{self.loss_fn}"
        
    def forward(self, text_code_pair_args):
        # text_code_pair_args: ids, attn_mask with "[CLS] <text> [SEP] <code> [SEP]"
        text_code_pair_embed = self.model(*text_code_pair_args).pooler_output # (batch, emb_size)
        # print("text_code_pair_embed.device =", text_code_pair_embed.device)
        # x = self.mlp(text_code_pair_embed)
        # print("x.device =", x.device)
        # x = self.sigmoid(x)
        return self.sigmoid(self.mlp(text_code_pair_embed))
#     def predict(self, q_and_c, **args):
#         queries_and_candidates = [] 
#         batch_size = args.get("batch_size", 32)
#         device = args.get("device") if torch.cuda.is_available() else "cpu"
#         dataset = TextCodePairDataset(q_and_c, tokenizer=self.tokenizer,
#                                       truncation=True, padding="max_length",
#                                       max_length=200, add_special_tokens=True,
#                                       return_tensors="pt")
#         dataloader = DataLoader(dataset, shuffle=False, 
#                                 batch_size=batch_size)
#         relevance_scores = []
#         pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="predicting relevance")
#         for step, batch in pbar:
#             with torch.no_grad():
#                 text_code_pair = (batch[0].to(device), batch[1].to(device))
#                 pred_reg_score = self(text_code_pair).squeeze().tolist()
#                 relevance_scores += pred_reg_score
#                 # if step == 5: break # DEBUG
#         return relevance_scores
    def val(self, valloader: DataLoader, epoch_i: int=0, epochs: int=0, device="cuda:0"):
        self.eval()
        batch_losses = []
        pbar = tqdm(enumerate(valloader), total=len(valloader), 
                    desc=f"val: epoch: {epoch_i+1}/{epochs} batch_loss: 0 loss: 0")
        tot = 0
        matches = 0
        for step, batch in pbar:
            with torch.no_grad():
                text_code_pair = (batch[0].to(device), batch[1].to(device))
                rel_label = batch[2].float().to(device) # 0 or 1.
                pred_probs = self(text_code_pair).squeeze()
                batch_loss = self.loss_fn(pred_probs, rel_label)
                batch_losses.append(batch_loss.item())
                tot += len(rel_label)
                matches += ((pred_probs > 0.06441).float() == rel_label).sum().item()
                acc = (matches/tot)
                pbar.set_description(f"val: epoch: {epoch_i+1}/{epochs} acc: {100*acc:.2f} batch_loss: {batch_loss:.5f} loss: {np.mean(batch_losses):.5f}")
                # if step == 5: break # DEBUG
        return np.mean(batch_losses)
    
    def fit(self, train_path: str, val_path: str, **args):
        thresh: float = 0.06441
        batch_size = args.get("batch_size", 32)
        epochs = args.get("epochs", 5)
        device_id = args.get("device_id", "cuda:0")
        device = device_id if torch.cuda.is_available() else "cpu"
        exp_name = args.get("exp_name", "experiment")
        os.makedirs(exp_name, exist_ok=True)
        save_path = os.path.join(exp_name, "model.pt")
        # store config info.
        self.config["batch_size"] = batch_size
        self.config["train_path"] = train_path
        self.config["device_id"] = device_id
        self.config["exp_name"] = exp_name
        self.config["val_path"] = val_path
        self.config["epochs"] = epochs
        
        config_path = os.path.join(exp_name, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f)
        print(f"saved config to {config_path}")
        print(f"model will be saved at {save_path}")
        print(f"moving model to {device}")
        self.to(device)
        trainset = RelevanceClassifierDataset(train_path, tokenizer=self.tokenizer,
                                              truncation=True, padding="max_length",
                                              max_length=200, add_special_tokens=True,
                                              return_tensors="pt", thresh=0.06441)
        valset = RelevanceClassifierDataset(val_path, tokenizer=self.tokenizer,
                                            truncation=True, padding="max_length",
                                            max_length=200, add_special_tokens=True,
                                            return_tensors="pt", thresh=0.06441)
        trainloader = DataLoader(trainset, shuffle=True, 
                                 batch_size=batch_size)
        valloader = DataLoader(valset, shuffle=False,
                               batch_size=batch_size)
        train_metrics = {
            "epochs": [],
            "summary": [],
        } 
        best_val_loss = 100
        for epoch_i in range(epochs):
            tot = 0
            matches = 0
            self.train()
            batch_losses = []
            pbar = tqdm(enumerate(trainloader), total=len(trainloader),
                        desc=f"train: epoch: {epoch_i+1}/{epochs} batch_loss: 0 loss: 0")
            for step, batch in pbar:      
                text_code_pair = (batch[0].to(device), batch[1].to(device))
                rel_label = batch[2].float().to(device)
                pred_probs = self(text_code_pair).squeeze()
                # print(true_reg_score)
                # print(pred_reg_score.device)
                batch_loss = self.loss_fn(pred_probs, rel_label)
                batch_loss.backward()
                self.optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                self.zero_grad()
                batch_losses.append(batch_loss.item())
                tot += len(rel_label)
                matches += ((pred_probs > 0.06441).float() == rel_label).sum().item()
                acc = (matches/tot)
                pbar.set_description(f"train: epoch: {epoch_i+1}/{epochs} acc: {100*acc:.2f} batch_loss: {batch_loss:.5f} loss: {np.mean(batch_losses):.5f}")
                # if step == 5: break # DEBUG
            # validate current model
            val_loss = self.val(valloader, epoch_i=epoch_i, 
                                epochs=epochs, device=device)
            if val_loss < best_val_loss:
                print(f"saving best model till now with val_loss: {val_loss} at {save_path}")
                best_val_loss = val_loss
                torch.save(self.state_dict(), save_path)
            train_metrics["epochs"].append({
                "train_batch_losses": batch_losses, 
                "train_loss": np.mean(batch_losses), 
                "val_loss": val_loss,
            })
        
        return train_metrics
    
    
class CodeBERTRelevanceRegressor(nn.Module):
    """
    finetune CodeBERT over CoNaLa mined pairs 
    for predicting relevance score using regression
    """
    def __init__(self, model_path: str="microsoft/codebert-base", 
                 tok_path: str="microsoft/codebert-base", **args):
        super(CodeBERTRelevanceRegressor, self).__init__()
        self.config = {}
        self.config["model_path"] = model_path
        self.config["tok_path"] = tok_path
        
        print(f"loading pretrained CodeBERT embedding model from {model_path}")
        start = time.time()
        self.model = RobertaModel.from_pretrained(model_path)
        print(f"loaded CodeBERT model in {(time.time()-start):.2f}s")
        print(f"loaded tokenizer files from {tok_path}")
        self.mlp = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
        self.tokenizer = RobertaTokenizer.from_pretrained(tok_path)
        # optimizer and loss.
        adam_eps = 1e-8
        lr = args.get("lr", 1e-5)
        self.config["lr"] = lr
        
        print(f"optimizer = AdamW(lr={lr}, eps={adam_eps})")
        self.optimizer = AdamW(self.parameters(), eps=adam_eps, lr=lr)
        self.config["optimizer"] = f"{self.optimizer}"
        
        self.loss_fn = nn.MSELoss()
        print(f"loss_fn = {self.loss_fn}")
        self.config["loss_fn"] = f"{self.loss_fn}"
        
    def forward(self, text_code_pair_args):
        # text_code_pair_args: ids, attn_mask with "[CLS] <text> [SEP] <code> [SEP]"
        text_code_pair_embed = self.model(*text_code_pair_args).pooler_output # (batch, emb_size)
        # print("text_code_pair_embed.device =", text_code_pair_embed.device)
        # x = self.mlp(text_code_pair_embed)
        # print("x.device =", x.device)
        # x = self.sigmoid(x)
        return self.sigmoid(self.mlp(text_code_pair_embed))
    
    def predict(self, q_and_c, **args):
        queries_and_candidates = [] 
        batch_size = args.get("batch_size", 32)
        device = args.get("device") if torch.cuda.is_available() else "cpu"
        dataset = TextCodePairDataset(q_and_c, tokenizer=self.tokenizer,
                                      truncation=True, padding="max_length",
                                      max_length=200, add_special_tokens=True,
                                      return_tensors="pt")
        dataloader = DataLoader(dataset, shuffle=False, 
                                batch_size=batch_size)
        relevance_scores = []
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="predicting relevance")
        for step, batch in pbar:
            with torch.no_grad():
                text_code_pair = (batch[0].to(device), batch[1].to(device))
                pred_reg_score = self(text_code_pair).squeeze().tolist()
                relevance_scores += pred_reg_score
                # if step == 5: break # DEBUG
        return relevance_scores

    def val(self, valloader: DataLoader, epoch_i: int=0, epochs: int=0, device="cuda:0"):
        self.eval()
        batch_losses = []
        pbar = tqdm(enumerate(valloader), total=len(valloader), 
                    desc=f"val: epoch: {epoch_i+1}/{epochs} batch_loss: 0 loss: 0")
        for step, batch in pbar:
            with torch.no_grad():
                text_code_pair = (batch[0].to(device), batch[1].to(device))
                true_reg_score = batch[2].to(device)
                pred_reg_score = self(text_code_pair).squeeze()
                batch_loss = self.loss_fn(pred_reg_score, true_reg_score)
                batch_losses.append(batch_loss.item())
                pbar.set_description(f"val: epoch: {epoch_i+1}/{epochs} batch_loss: {batch_loss:.5f} loss: {np.mean(batch_losses):.5f}")
                # if step == 5: break # DEBUG
        return np.mean(batch_losses)
    
    def fit(self, train_path: str, val_path: str, **args):
        batch_size = args.get("batch_size", 32)
        self.config["batch_size"] = batch_size
        epochs = args.get("epochs", 5)
        self.config["epochs"] = epochs
        device_id = args.get("device_id", "cuda:0")
        self.config["device_id"] = device_id
        device = device_id if torch.cuda.is_available() else "cpu"
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
        self.to(device)
        trainset = RelevanceRegressionDataset(train_path, tokenizer=self.tokenizer,
                                              truncation=True, padding="max_length",
                                              max_length=200, add_special_tokens=True,
                                              return_tensors="pt")
        valset = RelevanceRegressionDataset(val_path, tokenizer=self.tokenizer,
                                            truncation=True, padding="max_length",
                                            max_length=200, add_special_tokens=True,
                                            return_tensors="pt")
        trainloader = DataLoader(trainset, shuffle=True, 
                                 batch_size=batch_size)
        valloader = DataLoader(valset, shuffle=False,
                               batch_size=batch_size)
        train_metrics = {
            "epochs": [],
            "summary": [],
        } 
        best_val_loss = 100
        for epoch_i in range(epochs):
            self.train()
            batch_losses = []
            pbar = tqdm(enumerate(trainloader), total=len(trainloader),
                        desc=f"train: epoch: {epoch_i+1}/{epochs} batch_loss: 0 loss: 0")
            for step, batch in pbar:      
                text_code_pair = (batch[0].to(device), batch[1].to(device))
                true_reg_score = batch[2].to(device)
                pred_reg_score = self(text_code_pair).squeeze()
                # print(true_reg_score)
                # print(pred_reg_score.device)
                batch_loss = self.loss_fn(pred_reg_score, true_reg_score)
                batch_loss.backward()
                self.optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                self.zero_grad()
                batch_losses.append(batch_loss.item())
                pbar.set_description(f"train: epoch: {epoch_i+1}/{epochs} batch_loss: {batch_loss:.5f} loss: {np.mean(batch_losses):.5f}")
                # if step == 5: break # DEBUG
            # validate current model
            val_loss = self.val(valloader, epoch_i=epoch_i, 
                                         epochs=epochs, device=device)
            if val_loss < best_val_loss:
                print(f"saving best model till now with val_loss: {val_loss} at {save_path}")
                best_val_loss = val_loss
                torch.save(self.state_dict(), save_path)
            train_metrics["epochs"].append({
                "train_batch_losses": batch_losses, 
                "train_loss": np.mean(batch_losses), 
                "val_loss": val_loss,
            })
        
        return train_metrics
        
    
class CodeBERTripletNet(nn.Module):
    """ Class to 
    1) finetune CodeBERT in a late fusion setting using triplet margin loss.
    2) Evaluate metrics on unseen test set.
    3) 
    """
    def __init__(self, model_path: str="microsoft/codebert-base", 
                 tok_path: str="microsoft/codebert-base", **args):
        super(CodeBERTripletNet, self).__init__()
        self.config = {}
        self.config["model_path"] = model_path
        self.config["tok_path"] = tok_path
        
        margin = args.get("margin", 1)
        dist_fn_deg = args.get("dist_fn_deg", 2)
        self.config["margin"] = margin
        self.config["dist_fn_deg"] = dist_fn_deg
        print(f"loading pretrained CodeBERT embedding model from {model_path}")
        start = time.time()
        self.embed_model = RobertaModel.from_pretrained(model_path)
        print(f"loaded embedding model in {(time.time()-start):.2f}s")
        print(f"loaded tokenizer files from {tok_path}")
        self.tokenizer = RobertaTokenizer.from_pretrained(tok_path)
        # optimizer and loss.
        adam_eps = 1e-8
        lr = args.get("lr", 1e-5)
        self.config["lr"] = lr
        print(f"optimizer = AdamW(lr={lr}, eps={adam_eps})")
        self.optimizer = AdamW(self.parameters(), eps=adam_eps, lr=lr)
        print(f"loss_fn = TripletMarginLoss(margin={margin}, p={dist_fn_deg})")
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=dist_fn_deg)
        self.config["optimizer"] = f"{self.optimizer}"
        self.config["loss_fn"] = f"{self.loss_fn}"
        
    def forward(self, anchor_title, pos_snippet, neg_snippet):
        anchor_text_emb = self.embed_model(*anchor_title).pooler_output # get [CLS] token (batch, emb_size)
        pos_code_emb = self.embed_model(*pos_snippet).pooler_output # get [CLS] token (batch, emb_size)
        neg_code_emb = self.embed_model(*neg_snippet).pooler_output # get [CLS] token (batch, emb_size)
        
        return anchor_text_emb, pos_code_emb, neg_code_emb
        
    def val(self, valloader: DataLoader, epoch_i: int=0, epochs: int=0, device="cuda:0"):
        self.eval()
        val_acc = TripletAccuracy()
        batch_losses = []
        pbar = tqdm(enumerate(valloader), total=len(valloader), 
                    desc=f"val: epoch: {epoch_i+1}/{epochs} batch_loss: 0 loss: 0 acc: 0")
        for step, batch in pbar:
            with torch.no_grad():
                anchor_title = (batch[0].to(device), batch[1].to(device))
                pos_snippet = (batch[2].to(device), batch[3].to(device))
                neg_snippet = (batch[4].to(device), batch[5].to(device))
                anchor_text_emb, pos_code_emb, neg_code_emb = self(anchor_title, pos_snippet, neg_snippet)
                batch_loss = self.loss_fn(anchor_text_emb, pos_code_emb, neg_code_emb)
                val_acc.update(anchor_text_emb, pos_code_emb, neg_code_emb)
                batch_losses.append(batch_loss.item())
                pbar.set_description(f"val: epoch: {epoch_i+1}/{epochs} batch_loss: {batch_loss:.3f} loss: {np.mean(batch_losses):.3f} acc: {100*val_acc.get():.2f}")
                # if step == 5: break # DEBUG
        return val_acc.get(), np.mean(batch_losses)
        
    def encode_emb(self, text_or_snippets: List[str], mode: str="text", **args) -> list:
        """Note: our late fusion CodeBERT is a universal encoder for text and code, so the same function works for both."""
        device_id = args.get("device_id", "cuda:0")
        batch_size = args.get("batch_size", 32)
        use_tqdm = args.get("use_tqdm", False)
        
        device = device_id if torch.cuda.is_available() else "cpu"
        self.to(device)
        self.eval()
        
        if mode == "text":
            dataset = TextDataset(text_or_snippets, tokenizer=self.tokenizer,
                                  truncation=True, padding="max_length",
                                  max_length=100, add_special_tokens=True,
                                  return_tensors="pt")
        elif mode == "code":
            dataset = CodeDataset(text_or_snippets, tokenizer=self.tokenizer,
                                  truncation=True, padding="max_length",
                                  max_length=100, add_special_tokens=True,
                                  return_tensors="pt")
        else: raise TypeError("Unrecognized encoding mode")
        
        datalloader = DataLoader(dataset, shuffle=False, 
                                 batch_size=batch_size)
        pbar = tqdm(enumerate(datalloader), total=len(datalloader), 
                    desc=f"enocding {mode}", disable=not(use_tqdm))
        all_embeds = []
        for step, batch in pbar:
            with torch.no_grad():
                enc_args = (batch[0].to(device), batch[1].to(device))
                batch_embed = self.embed_model(*enc_args).pooler_output
                for embed in batch_embed: all_embeds.append(embed)
                # if step == 5: break # DEBUG
        # print(type(all_embeds[0]), len(all_embeds))
        return all_embeds
    
    def write_encode_emb_libsvm(self, text_or_snippets: List[str], 
                                path: str, mode: str="text", **args):
        """write the encoded embedding directly to a LIBSVM style text file."""
        device_id = args.get("device_id", "cuda:0")
        batch_size = args.get("batch_size", 32)
        use_tqdm = args.get("use_tqdm", False)
        
        device = device_id if torch.cuda.is_available() else "cpu"
        self.to(device)
        self.eval()
        file_ptr = open(path, "w")
        
        if mode == "text":
            dataset = TextDataset(text_or_snippets, tokenizer=self.tokenizer,
                                  truncation=True, padding="max_length",
                                  max_length=100, add_special_tokens=True,
                                  return_tensors="pt")
        elif mode == "code":
            dataset = CodeDataset(text_or_snippets, tokenizer=self.tokenizer,
                                  truncation=True, padding="max_length",
                                  max_length=100, add_special_tokens=True,
                                  return_tensors="pt")
        else: raise TypeError("Unrecognized encoding mode")
        datalloader = DataLoader(dataset, shuffle=False, 
                                 batch_size=batch_size)
        pbar = tqdm(enumerate(datalloader), total=len(datalloader), 
                    desc=f"enocding {mode}", disable=not(use_tqdm))
        for step, batch in pbar:
            with torch.no_grad():
                enc_args = (batch[0].to(device), batch[1].to(device))
                batch_embed = self.embed_model(*enc_args).pooler_output
                for embed in batch_embed:
                    file_ptr.write(str(embed)+"\n")
                # if step == 5: break # DEBUG
    def fit(self, train_path: str, val_path: str, **args):
        exp_name = args.get("exp_name", "experiment")
        device_id = args.get("device_id", "cuda:0")
        batch_size = args.get("batch_size", 32)
        epochs = args.get("epochs", 5)
        do_dynamic_negative_sampling = args.get("dynamic_negative_sampling", False)
        
        device = device_id if torch.cuda.is_available() else "cpu"
        save_path = os.path.join(exp_name, "model.pt")
        # create experiment folder.
        os.makedirs(exp_name, exist_ok=True)
        # save params to config file.
        self.config["batch_size"] = batch_size
        self.config["train_path"] = train_path
        self.config["device_id"] = device_id
        self.config["exp_name"] = exp_name
        self.config["val_path"] = val_path
        self.config["epochs"] = epochs
        self.config["dynamic_negative_sampling"] = do_dynamic_negative_sampling
        
        config_path = os.path.join(exp_name, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f)
        print(f"saved config to {config_path}")
        print(f"model will be saved at {save_path}")
        print(f"moving model to {device}")
        self.embed_model.to(device)
        trainset = TriplesDataset(train_path, tokenizer=self.tokenizer,
                                  truncation=True, padding="max_length",
                                  max_length=100, add_special_tokens=True,
                                  return_tensors="pt")
        valset = TriplesDataset(val_path, tokenizer=self.tokenizer,
                                truncation=True, padding="max_length",
                                max_length=100, add_special_tokens=True,
                                return_tensors="pt")
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
                if do_dynamic_negative_sampling:
                    batch = dynamic_negative_sampling(
                        self.embed_model, batch, 
                        model_name="codebert", 
                        device=device, k=1,
                    )
                anchor_title = (batch[0].to(device), batch[1].to(device))
                pos_snippet = (batch[2].to(device), batch[3].to(device))
                neg_snippet = (batch[4].to(device), batch[5].to(device))
                anchor_text_emb, pos_code_emb, neg_code_emb = self(anchor_title, pos_snippet, neg_snippet)
                batch_loss = self.loss_fn(anchor_text_emb, pos_code_emb, neg_code_emb)
                batch_loss.backward()
                self.optimizer.step()
                train_acc.update(anchor_text_emb, pos_code_emb, neg_code_emb)
                # scheduler.step()  # Update learning rate schedule
                self.zero_grad()
                batch_losses.append(batch_loss.item())
                pbar.set_description(f"train: epoch: {epoch_i+1}/{epochs} batch_loss: {batch_loss:.3f} loss: {np.mean(batch_losses):.3f} acc: {100*train_acc.get():.2f}")
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
    tok_path = os.path.join(os.path.expanduser("~"), "codebert-base-tok")
    print("creating model object")
    triplet_net = CodeBERTripletNet(tok_path=tok_path, **vars(args))
    print("commencing training")
    
    metrics = triplet_net.fit(train_path=args.train_path, batch_size=args.batch_size,
                              device_id=args.device_id, val_path=args.val_path, 
                              exp_name=args.exp_name, epochs=args.epochs,
                              dynamic_negative_sampling=args.dynamic_negative_sampling)
    metrics_path = os.path.join(args.exp_name, "train_metrics.json")
    
    print(f"saving metrics to {metrics_path}")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

def train_classifier(args):
    import os
    print("\x1b[33;1mtraining relevance classifier\x1b[0m")
    print("initializing model and tokenizer ..")
    tok_path = models.get_tok_path("codebert")
    print("creating model object")
    rel_classifier = CodeBERTRelevanceClassifier(tok_path=tok_path)
    print("commencing training")
    metrics = rel_classifier.fit(train_path=args.train_path, 
                                 device_id=args.device_id,
                                 val_path=args.val_path, 
                                 exp_name=args.exp_name,
                                 epochs=5)
    metrics_path = os.path.join(args.exp_name, "train_metrics.json")
    print(f"saving metrics to {metrics_path}")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
        
def train_regressor(args):
    import os
    print("\x1b[33;1mtraining relevance regressor\x1b[0m")
    print("initializing model and tokenizer ..")
    tok_path = models.get_tok_path("codebert")
    print("creating model object")
    rel_regressor = CodeBERTRelevanceRegressor(tok_path=tok_path)
    print("commencing training")
    metrics = rel_regressor.fit(train_path=args.train_path, 
                                device_id=args.device_id,
                                val_path=args.val_path, 
                                exp_name=args.exp_name,
                                epochs=5)
    metrics_path = os.path.join(args.exp_name, "train_metrics.json")
    print(f"saving metrics to {metrics_path}")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

def test_regressor(args):
    print("\x1b[33;1mtesting relevance regressor\x1b[0m")
    print("initializing model and tokenizer ..")
    tok_path = models.get_tok_path("codebert")
    print("creating model object")
    
    predict_path = args.predict_path
    stem, ext = os.path.splitext(predict_path)
    save_path = f"{stem}_rel_scores{ext}"
    if os.path.exists(save_path):
        print(f"""file \x1b[34;1m'{save_path}'\x1b[0m already exists and won't be overwritten. 
              Please delete the file if you are sure you don't need it to proceed""")
        return
    rel_regressor = CodeBERTRelevanceRegressor(tok_path=tok_path)
    with open(predict_path) as f:
        triples = json.load(f)
    q_and_c = []
    for item in triples:
        a = item["a"]
        n = item["n"]
        #  only calculate relevance scores for missing/invalid values (-1)
        if item["r_an"] == -1:
            q_and_c.append((a,n))
    q_and_c = q_and_c[:100]
    rel_scores = rel_regressor.predict(q_and_c=q_and_c,
                                       device_id=args.device_id,
                                       exp_name=args.exp_name,
                                       batch_size=32)
    i = 0
    print("len(rel_scores)=", len(rel_scores))
    print(rel_scores)
    for item in triples[:100]:
        a = item["a"]
        n = item["n"]
        #  only calculate relevance scores for missing/invalid values (-1)
        if item["r_an"] == -1:
            item["r_an"] = rel_scores[i]
            i += 1
    print(f"saving relevance predictions to {save_path}")
    with open(save_path, "w") as f:
        json.dump(triples, f, indent=4)

def test_retreival(args):
    print("initializing model and tokenizer ..")
    tok_path = os.path.join(os.path.expanduser("~"), "codebert-base-tok")
    device = args.device_id if torch.cuda.is_available() else "cpu"
    
    ckpt_path = os.path.join(args.exp_name, "model.pt")
    print(f"loading checkpoint (state dict) from {ckpt_path}")
    try: state_dict = torch.load(ckpt_path, map_location="cpu")
    except Exception as e: 
        state_dict = None
        print("Couldn't load state dict because:")
        print(e)
    
    print("creating model object")
    triplet_net = CodeBERTripletNet(tok_path=tok_path, **vars(args))
    if state_dict: 
        print(f"\x1b[32;1mloading state dict from {ckpt_path}\x1b[0m")
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
                                               batch_size=args.batch_size,
                                               use_tqdm=True, device_id=device)
            query_mat = torch.stack(query_mat)

            print(f"encoding {len(candidates)} candidates:")
            if setting == "code":
                cand_mat = triplet_net.encode_emb(candidates, mode="code", 
                                                  batch_size=args.batch_size,
                                                  use_tqdm=True, device_id=device)
                cand_mat = torch.stack(cand_mat)
            elif setting == "annot":
                cand_mat = triplet_net.encode_emb(candidates, mode="text", 
                                                  batch_size=args.batch_size,
                                                  use_tqdm=True, device_id=device)
                cand_mat = torch.stack(cand_mat)
            else:
                cand_mat_code = triplet_net.encode_emb(code_candidates, mode="code", 
                                                       batch_size=args.batch_size,
                                                       use_tqdm=True, device_id=device)
                cand_mat_annot = triplet_net.encode_emb(annot_candidates, mode="text",
                                                        batch_size=args.batch_size,
                                                        use_tqdm=True, device_id=device)
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
                    "CodeBERT_zero_shot", 
                    f"test_metrics_{dist_func}_{setting}.json"
                )
                os.makedirs("CodeBERT_zero_shot", exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f)

                
if __name__ == "__main__":
    args = get_args()
    if args.train_cls: # do relevance classification.
        train_classifier(args)
    if args.test_cls: # test relevance classification.
        test_classifier(args)
    if args.train_rel: # do regression.
        train_regressor(args)
    if args.test_rel: # test regression,
        test_regressor(args)
    if args.train: # finetune.
        main(args)
    if args.test: # setting in ['code', 'annot', 'code+annot']
        test_retreival(args)
    if args.test_ood: 
        print("creating model object")
        # instantiate model class.
        tok_path = get_tok_path("codebert")
        triplet_net = CodeBERTripletNet(tok_path=tok_path, **vars(args))
        test_ood_performance(
            triplet_net, model_name="codebert", args=args,
            query_paths=["query_and_candidates.json", "external_knowledge/queries.json", 
                         "data/queries_webquery.json", "data/queries_codesearchnet.json"],
            cand_paths=["candidate_snippets.json", "external_knowledge/candidates.json",
                        "data/candidates_webquery.json", "data/candidates_codesearchnet.json"], 
        )