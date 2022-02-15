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
from typing import Union
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer

# seed shit
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# get arguments
def get_args():
    parser = argparse.ArgumentParser("script to train (using triplet margin loss), evaluate and predict with the CodeBERT in Late Fusion configuration for Neural Code Search.")
    parser.add_argument("-tp", "--train_path", type=str, default="triples_train.json")
    parser.add_argument("-vp", "--val_path", type=str, default="triples_val.json")
    # parser.add_argument("-sp", "--save_path", type=str, default="triplet_CodeBERT.pt")
    parser.add_argument("-en", "--exp_name", type=str, default="triplet_CodeBERT")
    
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
        print(self.count, self.tot)
    
# TripletMarginWithDistanceLoss for custom design function.
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
                pbar.set_description(f"val: epoch: {epoch_i+1}/{epochs} batch_loss: {batch_loss:.3f} loss: {np.mean(batch_losses):.3f} acc: {100*train_acc.get():.2f}")
                if step == 5: break # DEBUG
                
        return val_acc.get(), np.mean(batch_losses)
        
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
        valset = TriplesDataset(train_path, tokenizer=self.tokenizer,
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
                if step == 5: break # DEBUG
            # validate current model
            val_acc, val_loss = self.val(valloader, epoch_i=epoch_i, 
                                         epochs=epochs, device=device)
            if val_acc > best_val_acc:
                print(f"saving best model till now with val_acc: {val_acc}")
                best_val_acc = val_acc
                model.save(save_path)
            train_metrics["epochs"].append({
                "train_batch_losses": batch_losses, 
                "train_loss": np.mean(batch_losses), 
                "train_acc": 100*train_acc.get(),
                "val_loss": val_loss,
                "val_acc": 100*val_acc,
            })
        
        return train_metrics
#     def predict(self):
#         self.tokenizer.
def main():
    import os
    args = get_args()
    print("initializing model and tokenizer ..")
    tok_path = os.path.join(os.path.expanduser("~"), "codebert-base-tok")
    print("creating model object")
    triplet_net = CodeBERTripletNet(tok_path=tok_path)
    print("commencing training")
    triplet_net.fit(train_path=args.train_path, 
                    val_path=args.val_path, 
                    exp_name=args.exp_name)
    
    
if __name__ == "__main__":
    main()