#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Atharva Naik (18CS10067)
# Neural bag of words baseline for text - code and text - text retrieval.
import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
from typing import Tuple, Union, List
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer


class TextCodePairDataset(Dataset):
    def __init__(self, path: str, tokenizer: Union[str, None, RobertaTokenizer]=None, **tok_args):
        super(TextCodePairDataset, self).__init__()
        self.data = json.load(open(path))
        self.tok_args = tok_args
        if isinstance(tokenizer, RobertaTokenizer):
            self.tokenizer = tokenizer
        elif isinstance(tokenizer, str):
            tokenizer = os.path.expanduser(tokenizer)
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
        label = self.data[i][2]
        if self.tokenizer:
            # special tokens are added by default.
            text = self.tokenizer(text, **self.tok_args)
            code = self.tokenizer(code, **self.tok_args)
            label = torch.as_tensor(label)
            return [text["input_ids"][0], code["input_ids"][0], label]
        else:
            return [text, code, label]
    

class NBowEncoder(nn.Module):
    def __init__(self, embedding):
        super(NBowEncoder, self).__init__()
        # initialize layers
        self.embedding = embedding # initialize embedding layer from CodeBERT.
        # self.softmax = nn.Softmax(dim=-1)
        # self.sigmoid = nn.Sigmoid()
    def forward(self, text_or_code: torch.Tensor) -> torch.Tensor:
        '''given batch_size x seq_len tensor of token ids, return mean pooled embedding of text (query/annotation) /code (snippet)'''
        embed = self.embedding(text_or_code) # batch_size x seq_len -> batch_size x seq_len x hidden_size  
        pooler_output = embed.mean(dim=1) # batch_size x seq_len x hidden_size -> batch_size x hidden_size  
    
        return pooler_output
    
def init_nbow_from_codebert():
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    nbow_encoder = NBowEncoder(model.embeddings.word_embeddings)
    
    return nbow_encoder


class SiameseWrapperNet(nn.Module):
    def __init__(self):
        super(SiameseWrapperNet, self).__init__()
        self.code_encoder = init_nbow_from_codebert()
        self.text_encoder = init_nbow_from_codebert()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text: torch.Tensor, code: torch.Tensor):
        code_enc = self.code_encoder(code)
        text_enc = self.text_encoder(text)
        dot_scores = (code_enc * text_enc).sum(-1)
        activations = self.sigmoid(dot_scores)
    
        return activations
    
    
def get_args():
    parser = argparse.ArgumentParser("script to train neural bag of words model using NL-PL pairs. task is to classify as negative/positive")
    parser.add_argument("-tp", "--train_path", type=str, default="triples/nl_code_pairs_train.json")
    parser.add_argument("-vp", "--val_path", type=str, default="triples/nl_code_pairs_val.json")
    parser.add_argument("-en", "--exp_name", type=str, default="nbow_siamese")
    parser.add_argument("-c", "--candidates_path", type=str, default="candidate_snippets.json")
    parser.add_argument("-q", "--queries_path", type=str, default="query_and_candidates.json")
    parser.add_argument("-dt", "--do_train", action="store_true")
    parser.add_argument("-dp", "--do_predict", action="store_true")
    parser.add_argument("-d", "--device_id", default="cuda:0", type=str)
    parser.add_argument("-bs", "--batch_size", default=32, type=int)
    parser.add_argument("-e", "--epochs", default=20, type=int)
    # parser.add_argument("-cp", "--ckpt_path", type=str, default="triplet_CodeBERT_rel_thresh/model.pt")
    return parser.parse_args()
    
def val(model, valloader, epoch_i=0, epochs=0, device="cpu"):
    model.eval()
    val_acc = 0
    val_tot = 0
    batch_losses = []
    loss_fn = nn.BCELoss
    pbar = tqdm(enumerate(valloader), total=len(valloader), 
                desc=f"val: epoch: {epoch_i+1}/{epochs} batch_loss: 0 loss: 0 acc: 0")
    for step, batch in pbar:
        with torch.no_grad():
            text = batch[0].to(device)
            code = batch[1].to(device)
            trues = batch[2].to(device)
            
            probs = model(text, code)
            batch_loss = loss_fn(probs, trues)
            val_acc += ((probs>0.5)*trues).sum().item()
            val_tot += len(trues)
            
            batch_losses.append(batch_loss.item())
            pbar.set_description(f"val: epoch: {epoch_i+1}/{epochs} batch_loss: {batch_loss:.3f} loss: {np.mean(batch_losses):.3f} acc: {100*val_acc.get():.2f}")
            if step == 5: break # DEBUG
                
    return val_acc/val_tot, np.mean(batch_losses)
    
def finetune(args):
    # config file to be saved.
    config = {}
    lr = args.get("lr", 1e-5)
    device = torch.device(args.get("device_id", "cuda:0"))
    loss_fn = nn.BCELoss()
    config["loss_fn"] = str(loss_fn)
    config["lr"] = lr
    
    exp_name = args.get("exp_name", "experiment")
    config["exp_name"] = exp_name
    os.makedirs(exp_name, exist_ok=True)
    # instantiate wrapper siamese net container.
    print("creating Siamese Network for finetuning")
    model = SiameseWrapperNet()
    model.to(device)
    print("instantiated network")
    # create AdamW optimizer.
    optimizer = AdamW(model.parameters(), eps=1e-8, lr=lr)
    config["optimizer"] = str(optimizer)
    # batch size, num epochs.
    batch_size = args.get("batch_size", 32)
    config["batch_size"] = batch_size
    epochs = args.get("epochs", 20)
    config["epochs"] = epochs
    # create train and val loaders.
    val_path = args['val_path']
    train_path = args['train_path']
    config["val_path"] = val_path
    config["train_path"] = train_path
    tok_args = {
        "max_length": 100,
        "truncation": True,
        "return_tensors": "pt",
        "padding": "max_length",
        "add_special_tokens": False,
    }
    config.update(tok_args)
    valset = TextCodePairDataset(val_path, tokenizer="~/codebert-base-tok", **tok_args)
    trainset = TextCodePairDataset(train_path, tokenizer="~/codebert-base-tok", **tok_args)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    valloader = DataLoader(valset, batch_size=32, shuffle=False)
    # save path.
    save_path = os.path.join(exp_name, 'model.pt')
    config_path = os.path.join(exp_name, "config.json")
    config["save_path"] = save_path
    print(f"saved config to {config_path}")
    print(f"model will be saved at {save_path}")
    print(f"moving model to {device}")
    # training metrics.
    train_metrics = {
        "epochs": [],
        "summary": [],
    } 
    train_acc = 0
    train_tot = 0
    best_val_acc = 0
    # save config info.
    print(config)
    with open(config_path, "w") as f:
        json.dump(config, f)
    # train-eval loop.
    for epoch_i in range(epochs):
        model.train()
        batch_losses = []
        pbar = tqdm(enumerate(trainloader), total=len(trainloader),
                    desc=f"train: epoch: {epoch_i+1}/{epochs} batch_loss: 0 loss: 0 acc: 0")
        train_acc = 0
        for step, batch in pbar:
            text = batch[0].to(device)
            code = batch[1].to(device)
            trues = batch[2].to(device).float()
            
            probs = model(text, code)
            batch_loss = loss_fn(probs, trues)
            batch_loss.backward()
            optimizer.step()
            
            train_acc += ((probs>0.5)*trues).sum().item()
            train_tot += len(trues)
            # scheduler.step()  # Update learning rate schedule
            optimizer.zero_grad()
            batch_losses.append(batch_loss.item())
            pbar.set_description(f"train: epoch: {epoch_i+1}/{epochs} batch_loss: {batch_loss:.3f} loss: {np.mean(batch_losses):.3f} acc: {100*train_acc.get():.2f}")
            if step == 5: break # DEBUG
        # validate current model
        val_acc, val_loss = val(model, valloader, epoch_i=epoch_i, 
                                epochs=epochs, device=device)
        if val_acc > best_val_acc:
            print(f"saving best model till now with val_acc: {val_acc} at {save_path}")
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
        
        train_metrics["epochs"].append({
            "train_batch_losses": batch_losses, 
            "train_loss": np.mean(batch_losses), 
            "train_acc": 100*train_acc,
            "val_loss": val_loss,
            "val_acc": 100*val_acc,
        })

    return train_metrics
        
    
def main():
    args = get_args()
    if args.do_train:
        train_metrics = finetune(vars(args))
        train_metrics_path = os.path.join(args.exp_name, "train_metrics.json")
        with open(train_metrics_path, "w") as f:
            json.dump(train_metrics, f)
    
    
if __name__ == "__main__":
    main()