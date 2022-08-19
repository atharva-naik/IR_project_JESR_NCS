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
from baselines.CNN import CNNEncoder
from baselines.RNN import RNNEncoder
from typing import Tuple, Union, List
from baselines.nbow import NBowEncoder
from baselines import test_ood_performance
from sklearn.metrics import ndcg_score as NDCG
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from sklearn.metrics import label_ranking_average_precision_score as MRR

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
        
        
class TextCodePairDataset(Dataset):
    def __init__(self, path: str, tokenizer: Union[str, None, RobertaTokenizer]=None, **tok_args):
        super(TextCodePairDataset, self).__init__()
        data = json.load(open(path))
        code_dropped = 0
        text_dropped = 0
        self.data = []
        for text, code, label in data:
            if code.strip() == "":
                code_dropped += 1; continue
            else: self.data.append((text, code, label))
            if text.strip() == "":
                text_dropped += 1; continue
            else: self.data.append((text, code, label))           
        print(f"\x1b[1m{path}:\x1b[0m n_dropped(code)=", code_dropped)
        print(f"\x1b[1m{path}:\x1b[0m n_dropped(text)=", text_dropped)
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
            return [text["input_ids"][0], code["input_ids"][0], label, 
                    text["attention_mask"][0], code["attention_mask"][0]]
        else:
            return [text, code, label]
    
def init_codebert_embed():
    import time
    s = time.time()
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    print(f"initialized CodeBERT embedding layer in {time.time()-s}s")
    # nbow_encoder = NBowEncoder(model.embeddings.word_embeddings)
    return model.embeddings.word_embeddings


class SiameseWrapperNet(nn.Module):
    def __init__(self, enc_type="nbow", init_codebert=False, device="cpu"):
        super(SiameseWrapperNet, self).__init__()
        self.enc_type = enc_type
        if enc_type == "nbow":
            codebert_embed = init_codebert_embed()
            self.code_encoder = NBowEncoder(codebert_embed)
            self.text_encoder = NBowEncoder(codebert_embed)
        elif enc_type == "CNN":
            if init_codebert:
                embeddings = init_codebert_embed()
            else:
                embeddings = None
            self.code_encoder = CNNEncoder(embeddings)
            self.text_encoder = CNNEncoder(embeddings)
            # self.loss_fn = nn.BCELoss() # nn.CrossEntropyLoss()
        elif enc_type == "RNN":
            if init_codebert:
                embeddings = init_codebert_embed()
            else:
                embeddings = None
            self.code_encoder = RNNEncoder(embeddings, device=device)
            self.text_encoder = RNNEncoder(embeddings, device=device)
        else:
            raise TypeError(f"no implementation found for encoder of type '{enc_type}'")
        self.loss_fn = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text: torch.Tensor, code: torch.Tensor, 
                text_masks: Union[torch.Tensor, None]=None, 
                code_masks: Union[torch.Tensor, None]=None):
        if self.enc_type == "nbow":
            code_enc = self.code_encoder(code)
            text_enc = self.text_encoder(text)
        else:
            code_enc = self.code_encoder(code, code_masks)
            text_enc = self.text_encoder(text, text_masks)
        # print("code_enc:", code_enc)
        # print("text_enc:", text_enc)
        dot_scores = (code_enc * text_enc).sum(-1)
        # print("dot_scores:", dot_scores)
        Z = self.sigmoid(dot_scores)
        # print("Z:", Z)
        return Z
    
    def encode(self, text_or_snippets: List[str], 
               mode: str="text", **args):
        """Note: our late fusion CodeBERT is a universal encoder for text and code, so the same function works for both."""
        batch_size = args.get("batch_size", 48)
        device = torch.device(
            args.get("device_id", "cuda:0")
        )
        use_tqdm = args.get("use_tqdm", False)
        enc_type = args.get("enc_type", "nbow")
        self.to(device)
        self.eval()
        if mode == "text":
            dataset = TextDataset(text_or_snippets, 
                                  tokenizer=os.path.expanduser("~/codebert-base-tok"),
                                  truncation=True, padding="max_length",
                                  max_length=100, add_special_tokens=True,
                                  return_tensors="pt")
        elif mode == "code":
            dataset = CodeDataset(text_or_snippets, 
                                  tokenizer=os.path.expanduser("~/codebert-base-tok"),
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
                if enc_type == "nbow":
                    batch_ids = batch[0].to(device)
                    if mode == "code": batch_embed = self.code_encoder(batch_ids)
                    if mode == "text": batch_embed = self.text_encoder(batch_ids)
                else:
                    batch_ids = batch[0].to(device)
                    attn_masks = batch[1].to(device)
                    if mode == "code": batch_embed = self.code_encoder(batch_ids, attn_masks)
                    if mode == "text": batch_embed = self.text_encoder(batch_ids, attn_masks)
                for embed in batch_embed: all_embeds.append(embed)
                # if step == 5: break # DEBUG
        # print(type(all_embeds[0]), len(all_embeds))
        return all_embeds
    
    
def recall_at_k(actual, predicted, k: int=10):
    rel = 0
    tot = 0
    for act_list, pred_list in zip(actual, predicted):
        for i in act_list:
            tot += 1
            if i in pred_list[:k]: rel += 1
                
    return rel/tot
    
def get_args():
    parser = argparse.ArgumentParser("script to train neural bag of words model using NL-PL pairs. task is to classify as negative/positive")
    parser.add_argument("-tp", "--train_path", type=str, default="triples/nl_code_pairs_train.json", help="path to trainset")
    parser.add_argument("-vp", "--val_path", type=str, default="triples/nl_code_pairs_val.json", help="path to valset")
    parser.add_argument("-c", "--candidates_path", type=str, default="candidate_snippets.json", help="path to candidates JSON for evaluation")
    parser.add_argument("-q", "--queries_path", type=str, default="query_and_candidates.json", help="path to queries JSON for evaluation")
    parser.add_argument("-en", "--exp_name", type=str, default="nbow_siamese", help="name to be used for the experiment folder")
    parser.add_argument("-d", "--device_id", default="cuda:0", type=str, help="cuda device name (e.g. 'cuda:0') to be used.")
    parser.add_argument("-enc", "--enc_type", type=str, default="nbow", help="the encoder architecture to be used: RNN, CNN, n-BOW")
    parser.add_argument("-ic", "--init_codebert", action="store_true", help="whether to use CodeBERT embedding layer weights")
    parser.add_argument("-ood", "--do_ood_test", action="store_true", help="do out of distribution testing across 4 datasets")
    parser.add_argument("-bs", "--batch_size", default=32, type=int, help="batch size during training and evaluation")
    parser.add_argument("-dp", "--do_predict", action="store_true", help="do prediction/evaluation")
    parser.add_argument("-dt", "--do_train", action="store_true", help="do training on dataset")
    parser.add_argument("-e", "--epochs", default=20, type=int, help="no. of training epochs")

    return parser.parse_args()
    
def val(model, valloader, epoch_i=0, 
        epochs=0, device="cpu", 
        enc_type="nbow"):
    model.eval()
    val_acc = 0
    val_tot = 0
    batch_losses = []
    pbar = tqdm(enumerate(valloader), total=len(valloader), 
                desc=f"val: epoch: {epoch_i+1}/{epochs} batch_loss: 0 loss: 0 acc: 0")
    for step, batch in pbar:
        with torch.no_grad():
            text = batch[0].to(device)
            code = batch[1].to(device)
            trues = batch[2].to(device).float()
            if enc_type == "nbow":
                probs = model(text, code)
            else:
                code_masks = batch[3].to(device)
                text_masks = batch[4].to(device)
                probs = model(text, code, text_masks, code_masks)
            batch_loss = model.loss_fn(probs, trues)
            # DEBUG: change later
            val_acc += ((probs>0.5) == trues).sum().item()
            val_tot += len(trues)
            
            batch_losses.append(batch_loss.item())
            pbar.set_description(f"val: epoch: {epoch_i+1}/{epochs} batch_loss: {batch_loss:.3f} loss: {np.mean(batch_losses):.3f} acc: {100*(val_acc/val_tot):.2f}")
            # if step == 5: break # DEBUG  
    return val_acc/val_tot, np.mean(batch_losses)
    
def get_tok_path(model_name: str) -> str:
    assert model_name in ["codebert", "graphcodebert"]
    if model_name == "codebert":
        tok_path = os.path.expanduser("~/codebert-base-tok")
        if not os.path.exists(tok_path):
            tok_path = "microsoft/codebert-base"
    elif model_name == "graphcodebert":
        tok_path = os.path.expanduser("~/graphcodebert-base-tok")
        if not os.path.exists(tok_path):
            tok_path = "microsoft/grapcodebert-base"
            
    return tok_path
    
def test_retreival(args):
    print("initializing model and tokenizer ..")
    #     tok_path = os.path.join(
    #         os.path.expanduser("~"), 
    #         "codebert-base-tok"
    #     )
    tok_path = get_tok_path("codebert")
    device = torch.device(args.device_id if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(args.exp_name, "model.pt")
    print(f"loading checkpoint (state dict) from {ckpt_path}")
    try: state_dict = torch.load(ckpt_path)
    except Exception as e: 
        state_dict = None; print(e)
    
    print("creating model object")
    model = SiameseWrapperNet(
        device=device,
        enc_type=args.enc_type,
        init_codebert=args.init_codebert,
    )
    if state_dict: model.load_state_dict(state_dict)
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
            query_mat = model.encode(queries, mode="text", 
                                     use_tqdm=True, device_id=device,
                                     enc_type=args.enc_type)
            query_mat = torch.stack(query_mat)

            print(f"encoding {len(candidates)} candidates:")
            if setting == "code":
                cand_mat = model.encode(candidates, mode="code", 
                                        use_tqdm=True, device_id=device,
                                        enc_type=args.enc_type)
                cand_mat = torch.stack(cand_mat)
            elif setting == "annot":
                cand_mat = model.encode(candidates, mode="text", 
                                        use_tqdm=True, device_id=device,
                                        enc_type=args.enc_type)
                cand_mat = torch.stack(cand_mat)
            else:
                cand_mat_code = model.encode(code_candidates, mode="code", 
                                             use_tqdm=True, device_id=device,
                                             enc_type=args.enc_type)
                cand_mat_annot = model.encode(annot_candidates, mode="text", 
                                              use_tqdm=True, device_id=device,
                                              enc_type=args.enc_type)
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
            # elif mode == "joint_cls": scores = model.joint_classify(queries, candidates)
            doc_ranks = scores.argsort(axis=1)
            if dist_func == "inner_prod":
                doc_ranks = doc_ranks.flip(dims=[1])
            label_ranks, recall_at_ = [], []
            avg_rank, avg_best_rank, M, N = 0, 0, 0, 0
            lrap_GT = np.zeros((len(queries), len(candidates)))
            
            for i in range(1,10+1):
                recall_at_.append(recall_at_k(labels, doc_ranks.tolist(), k=5*i))
            for i in range(len(labels)):
                for j in labels[i]:
                    lrap_GT[i][j] = 1
            for i, rank_list in enumerate(doc_ranks):
                rank_list = rank_list.tolist()
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
    
def finetune(args):
    # config file to be saved.
    config = {}
    lr = args.get("lr", 1e-3)
    enc_type = args.get("enc_type", "nbow")
    init_codebert = args.get("init_codebert", False)
    device = torch.device(args.get("device_id", "cuda:0"))
    config["lr"] = lr
    
    exp_name = args.get("exp_name", "experiment")
    config["exp_name"] = exp_name
    os.makedirs(exp_name, exist_ok=True)
    # instantiate wrapper siamese net container.
    print("creating Siamese Network for finetuning")
    model = SiameseWrapperNet(
        enc_type=enc_type, 
        init_codebert=init_codebert,
        device=device,
    )
    config["loss_fn"] = repr(model.loss_fn)
    print(model)
    print("\x1b[34;1mcode_encoder:\x1b[0m")
    print(model.code_encoder)
    print("\x1b[34;1mtext_encoder:\x1b[0m")
    print(model.text_encoder)
    model.to(device)
    print("instantiated network")
    # create AdamW optimizer.
    optimizer = AdamW(model.parameters(), eps=1e-8, lr=lr)
    config["optimizer"] = repr(optimizer)
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
#     for i, item in tqdm(enumerate(trainset), total=len(trainset)):
#         if len(item[0]) != 100:
#             print(f"\x1b[31;1m{i}\x1b[0m", len(item), print(item))
#             exit()
#         if len(item[1]) != 100:
#             print(f"\x1b[31;1m{i}\x1b[0m", len(item), print(item))
#             exit()
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
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
        train_tot = 0
        for step, batch in pbar:
            text = batch[0].to(device)
            code = batch[1].to(device)
            trues = batch[2].to(device).float()
            if enc_type != "nbow":
                text_masks = batch[3].to(device) 
                code_masks = batch[4].to(device)
                probs = model(text, code, text_masks, code_masks)
            else:
                probs = model(text, code)
            batch_loss = model.loss_fn(probs, trues)
            batch_loss.backward()
            optimizer.step()
            # print(probs)
            train_acc += ((probs>0.5) == trues).sum().item()
            train_tot += len(trues)
            # scheduler.step()  # Update learning rate schedule
            optimizer.zero_grad()
            batch_losses.append(batch_loss.item())
            pbar.set_description(f"train: epoch: {epoch_i+1}/{epochs} batch_loss: {batch_loss:.3f} loss: {np.mean(batch_losses):.3f} acc: {100*(train_acc/train_tot):.2f}")
            # if step == 5: break # DEBUG
        # validate current model
        val_acc, val_loss = val(model, valloader, epoch_i=epoch_i, 
                                epochs=epochs, device=device, 
                                enc_type=enc_type)
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
    if args.do_predict:
        test_retreival(args)
    if args.do_ood_test:
        print("doing ood testing!")
        tok_path = get_tok_path("codebert")
        ckpt_path = os.path.join(args.exp_name, "model.pt")
        print(f"loading checkpoint (state dict) from {ckpt_path}")
        try: state_dict = torch.load(ckpt_path, map_location="cpu")
        except Exception as e: 
            state_dict = None; print(e)

        print("creating model object")
        model = SiameseWrapperNet(
            device=args.device_id,
            enc_type=args.enc_type, 
            init_codebert=args.init_codebert,
        )
        if state_dict: model.load_state_dict(state_dict)
        print(f"loading candidates from {args.candidates_path}")
        test_ood_performance(model, query_paths=["query_and_candidates.json", "external_knowledge/queries.json", 
                                                 "data/queries_webquery.json", "data/queries_codesearchnet.json"],
                             cand_paths=["candidate_snippets.json", "external_knowledge/candidates.json", 
                                         "data/candidates_webquery.json", "data/candidates_codesearchnet.json"], args=args)
    
if __name__ == "__main__":
    main()