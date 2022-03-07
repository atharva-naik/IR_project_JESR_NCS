#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from sklearn.metrics import label_ranking_average_precision_score as MRR

# seed shit
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# get arguments
def get_args():
    parser = argparse.ArgumentParser("script to train (using triplet margin loss), evaluate and predict with the CodeBERT in Late Fusion configuration for Neural Code Search.")
    parser.add_argument("-tp", "--train_path", type=str, default="triples_rel_thresh_train.json")
    parser.add_argument("-vp", "--val_path", type=str, default="triples_rel_thresh_val.json")
    parser.add_argument("-p", "--predict", action="store_true")
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-en", "--exp_name", type=str, default="triplet_CodeBERT_re_thresh")
    # parser.add_argument("-cp", "--ckpt_path", type=str, default="triplet_CodeBERT_rel_thresh/model.pt")
    parser.add_argument("-q", "--queries_path", type=str, default="query_and_candidates.json")
    parser.add_argument("-c", "--candidates_path", type=str, default="candidate_snippets.json")
    
    return parser.parse_args()
    
# triplet accuracy model.
class TripletAccuracy:
    def __init__(self):
        self.pdist = nn.PairwiseDistance()
        self.reset()
        
    def reset(self):
        self.count = 0
        self.tot = 0
        
    def get(self):
        return self.count/self.tot
        
    def update(self, anchor, pos, neg):
        pos = self.pdist(anchor, pos)
        neg = self.pdist(anchor, neg)
        self.count += torch.as_tensor((neg-pos)>0).sum().item()
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
        
        
class TextCodePairDataset(Dataset):
    def __init__(self, texts: str, codes: str, 
                 tokenizer: Union[str, None, RobertaTokenizer]=None, 
                 **tok_args):
        super(TextCodePairDataset, self).__init__()
        self.data = [(text, code) for text, code in zip(texts, codes)]
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
        
    def encode_emb(self, text_or_snippets: List[str], mode: str="text", **args):
        """Note: our late fusion CodeBERT is a universal encoder for text and code, so the same function works for both."""
        batch_size = args.get("batch_size", 48)
        device_id = args.get("device_id", "cuda:0")
        device = torch.device(device_id)
        use_tqdm = args.get("use_tqdm", False)
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
        
    def joint_classify(self, text_snippets: List[str], 
                       code_snippets: List[str], **args):
        """The usual joint encoding setup of CodeBERT (similar to NLI)"""
        batch_size = args.get("batch_size", 48)
        device_id = args.get("device_id", "cuda:0")
        device = torch.device(device_id)
        use_tqdm = args.get("use_tqdm", False)
        self.to(device)
        self.eval()
        
        dataset = TextCodePairDataset(text_snippets, code_snippets, 
                                      tokenizer=self.tokenizer, truncation=True, 
                                      padding="max_length", max_length=100, 
                                      add_special_tokens=True, return_tensors="pt")
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
        
    def fit(self, train_path: str, val_path: str, **args):
        batch_size = args.get("batch_size", 48)
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

    
def main():
    import os
    args = get_args()
    print("initializing model and tokenizer ..")
    tok_path = os.path.join(os.path.expanduser("~"), "codebert-base-tok")
    print("creating model object")
    triplet_net = CodeBERTripletNet(tok_path=tok_path)
    if args.train:
        print("commencing training")
        metrics = triplet_net.fit(train_path=args.train_path, 
                                  val_path=args.val_path, 
                                  exp_name=args.exp_name)
        metrics_path = os.path.join(args.exp_name, "train_metrics.json")
        print(f"saving metrics to {metrics_path}")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
    if args.predict:
        model_path = os.path.join(args.exp_name, "model.pt")
        print(model_path)
        
def test_retreival():
    import os
    import json
    args = get_args()
    print("initializing model and tokenizer ..")
    tok_path = os.path.join(os.path.expanduser("~"), "codebert-base-tok")
    
    ckpt_path = os.path.join(args.exp_name, "model.pt")
    metrics_path = os.path.join(args.exp_name, "test_metrics.json")
    print(f"loading checkpoint (state dict) from {ckpt_path}")
    try: state_dict = torch.load(ckpt_path)
    except Exception as e: 
        state_dict = None; print(e)
    
    print("creating model object")
    triplet_net = CodeBERTripletNet(tok_path=tok_path)
    if state_dict: triplet_net.load_state_dict(state_dict)
    print(f"loading candidates from {args.candidates_path}")
    candidates = json.load(open(args.candidates_path))
    
    print(f"loading queries from {args.queries_path}")
    queries_and_cand_labels = json.load(open(args.queries_path))
    queries = [i["query"] for i in queries_and_cand_labels]
    labels = [i["docs"] for i in queries_and_cand_labels]
    
    mode = "l2_dist"
    if mode in ["l2_dist", "inner_prod"]:
        print(f"encoding {len(queries)} queries:")
        query_mat = triplet_net.encode_emb(queries, mode="text", use_tqdm=True)
        query_mat = torch.stack(query_mat)

        print(f"encoding {len(candidates)} candidates:")
        cand_mat = triplet_net.encode_emb(candidates, mode="code", use_tqdm=True)
        cand_mat = torch.stack(cand_mat)
    # print(query_mat.shape, cand_mat.shape)
    if mode == "inner_prod": scores = query_mat @ cand_mat.T
    elif mode == "l2_dist": scores = torch.cdist(query_mat, cand_mat, p=2)
    elif mode == "joint_cls": scores = triplet_net.joint_classify(queries, candidates)
    doc_ranks = scores.argsort(axis=1)
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
        if mode == "inner_prod": rank_list = rank_list.tolist()[::-1]
        elif mode == "l2_dist": rank_list = rank_list.tolist()
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
    if mode == "inner_prod":
        # -scores for distance based scores, no - for innert product based scores.
        mrr = MRR(lrap_GT, scores.cpu().numpy())
    elif mode == "l2_dist":
        # -scores for distance based scores, no - for innert product based scores.
        mrr = MRR(lrap_GT, -scores.cpu().numpy())
    metrics["mrr"] = mrr
    print("MRR (LRAP):", mrr)
    if not os.path.exists(args.exp_name):
        print("missing experiment folder: assuming zero-shot setting")
        metrics_path = os.path.join(
            "CodeBERT_zero_shot", 
            "test_metrics.json"
        )
        os.makedirs("CodeBERT_zero_shot", exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
#     with open("pred_cand_ranks.json", "w") as f:
#         json.dump(label_ranks, f, indent=4)
if __name__ == "__main__":
    # main() 
    test_retreival()