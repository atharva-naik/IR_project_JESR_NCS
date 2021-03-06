#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Atharva Naik (18CS10067)
import os
import json
import time
import torch
import random
import argparse
import numpy as np
from typing import *
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
# load metrics.
from models.unixcoder import UniXcoder
from sklearn.metrics import ndcg_score as NDCG
from models.metrics import TripletAccuracy, recall_at_k 
from models import test_ood_performance, dynamic_negative_sampling
from sklearn.metrics import label_ranking_average_precision_score as MRR

# set logging level of transformers.
import transformers
transformers.logging.set_verbosity_error()
# seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# get arguments
def get_args():
    parser = argparse.ArgumentParser("script to train (using triplet margin loss), evaluate and predict with the UniXcoder in Late Fusion configuration for Neural Code Search.")
    parser.add_argument("-pp", "--predict_path", type=str, default="triples/triples_train.json", help="path to data for prediction of regression scores")
    parser.add_argument("-tp", "--train_path", type=str, default="triples/triples_train.json", help="path to training triplet data")
    parser.add_argument("-vp", "--val_path", type=str, default="triples/triples_test.json", help="path to validation triplet data")
    parser.add_argument("-c", "--candidates_path", type=str, default="candidate_snippets.json", help="path to candidates (to test retrieval)")
    parser.add_argument("-q", "--queries_path", type=str, default="query_and_candidates.json", help="path to queries (to test retrieval)")
    parser.add_argument("-en", "--exp_name", type=str, default="UniXcoder_rel_thresh", help="experiment name (will be used as folder name)")
    parser.add_argument("-d", "--device_id", type=str, default="cpu", help="device string (GPU) for doing training/testing")
    parser.add_argument("-lr", "--lr", type=float, default=1e-5, help="learning rate for training (defaults to 1e-5)")
    parser.add_argument("-te", "--test", action="store_true", help="flag to do testing")
    parser.add_argument("-t", "--train", action="store_true", help="flag to do training")
    parser.add_argument("-too", "--test_ood", action="store_true", help="flat to do ood testing")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="no. of epochs")
    parser.add_argument("-dns", "--dynamic_negative_sampling", action="store_true", 
                        help="do dynamic negative sampling at batch level")
    parser.add_argument("-sip", "--sim_intents_path", type=str, default=None, 
                        help="path to dictionary containing similar intents corresponding to a given intent")
    parser.add_argument("-pcp", "--perturbed_codes_path", type=str, default=None, 
                        help="path to dictionary containing AST perturbed codes corresponding to a given code")
    parser.add_argument("-ast", "--use_AST", action="store_true", help="use AST perturbed negative samples")
    parser.add_argument("-idns", "--intent_level_dynamic_sampling", action="store_true", 
                        help="dynamic sampling based on similar intents")
    # parser.add_argument("-cp", "--ckpt_path", type=str, default="UniXcoder_rel_thresh/model.pt")
    return parser.parse_args()

# TripletMarginWithDistanceLoss for custom design function.
class CodeDataset(Dataset):
    def __init__(self, code_snippets: str, model, **tok_args):
        super(CodeDataset, self).__init__()
        self.data = code_snippets
        self.model_ptr = model
        self.tok_args = tok_args
    
    def __len__(self):
        return len(self.data)
    
    def proc_code(self, code: str):
        code = " ".join(code.split("\n")).strip()
        return code
    
    def __getitem__(self, i: int):
        code = self.proc_code(self.data[i])
        if self.model_ptr:
            # special tokens are added by default.
            input_ids = self.model_ptr.tokenize([code], **self.tok_args)[0]          
            return torch.tensor(input_ids)
        else: return [code]
        
        
class TextDataset(Dataset):
    def __init__(self, texts: str, model=None, **tok_args):
        super(TextDataset, self).__init__()
        self.data = texts
        self.tok_args = tok_args
        self.model_ptr = model

    def __len__(self):
        return len(self.data)
    
    def proc_text(self, text: str):
        text = " ".join(text.split("\n"))
        text = " ".join(text.split()).strip()
        return text
    
    def __getitem__(self, i: int):
        text = self.proc_text(self.data[i])
        if self.model_ptr:
            # special tokens are added by default.
            input_ids = self.model_ptr.tokenize([text], **self.tok_args)[0]
            return torch.tensor(input_ids)
        else:
            return [text]

    
class TriplesDataset(Dataset):
    def __init__(self, path: str, model=None, **tok_args):
        super(TriplesDataset, self).__init__()
        self.data = json.load(open(path))
        self.tok_args = tok_args
        self.model_ptr = model
        
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
        # print(anchor)
        if self.model_ptr:
            # special tokens are added by default.
            anchor = self.model_ptr.tokenize([anchor], **self.tok_args)[0]
            pos = self.model_ptr.tokenize([pos], **self.tok_args)[0]
            neg = self.model_ptr.tokenize([neg], **self.tok_args)[0]
            # print(anchor)
            return [torch.tensor(anchor), 
                    torch.tensor(pos), 
                    torch.tensor(neg)]
        else:
            return [anchor, pos, neg]      
        
    
class UniXcoderTripletNet(nn.Module):
    """ Class to 
    1) finetune UniXcoder in a late fusion setting using triplet margin loss.
    2) Evaluate metrics on unseen test set.
    3) 
    """
    def __init__(self, model_path: str="microsoft/unixcoder-base", **args):
        super(UniXcoderTripletNet, self).__init__()
        self.config = {}
        self.config["model_path"] = model_path
        
        margin = args.get("margin", 1)
        dist_fn_deg = args.get("dist_fn_deg", 2)
        self.config["margin"] = margin
        self.config["dist_fn_deg"] = dist_fn_deg
        print(f"loading pretrained UniXcoder embedding model from {model_path}")
        start = time.time()
        self.embed_model = UniXcoder(model_path, tok_path="~/unixcoder-base-tok")
        print(f"loaded embedding model in {(time.time()-start):.2f}s")
        # print(f"loaded tokenizer files from {tok_path}")
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
        _,anchor_text_emb = self.embed_model(anchor_title)
        _,pos_code_emb = self.embed_model(pos_snippet)
        _,neg_code_emb = self.embed_model(neg_snippet)
        
        return anchor_text_emb, pos_code_emb, neg_code_emb
        
    def val(self, valloader: DataLoader, epoch_i: int=0, 
            epochs: int=0, device="cuda:0"):
        self.eval()
        val_acc = TripletAccuracy()
        batch_losses = []
        pbar = tqdm(enumerate(valloader), total=len(valloader), 
                    desc=f"val: epoch: {epoch_i+1}/{epochs} batch_loss: 0 loss: 0 acc: 0")
        for step, batch in pbar:
            with torch.no_grad():
                anchor_title = batch[0].to(device)
                pos_snippet = batch[1].to(device)
                neg_snippet = batch[2].to(device)
                anchor_text_emb, pos_code_emb, neg_code_emb = self(anchor_title, pos_snippet, neg_snippet)
                batch_loss = self.loss_fn(anchor_text_emb, pos_code_emb, neg_code_emb)
                val_acc.update(anchor_text_emb, pos_code_emb, neg_code_emb)
                batch_losses.append(batch_loss.item())
                pbar.set_description(f"val: epoch: {epoch_i+1}/{epochs} batch_loss: {batch_loss:.3f} loss: {np.mean(batch_losses):.3f} acc: {100*val_acc.get():.2f}")
                # if step == 5: break # DEBUG
        return val_acc.get(), np.mean(batch_losses)
        
    def encode_emb(self, text_or_snippets: List[str], mode: str="text", **args):
        """Note: our late fusion UniXcoder is a universal encoder for text and code, so the same function works for both."""
        device_id = args.get("device_id", "cuda:0")
        batch_size = args.get("batch_size", 32)
        use_tqdm = args.get("use_tqdm", False)
        
        device = device_id if torch.cuda.is_available() else "cpu"
        self.to(device)
        self.eval()
        
        if mode == "text":
            dataset = TextDataset(text_or_snippets, model=self.embed_model, 
                                  max_length=100, padding=True)
        elif mode == "code":
            dataset = CodeDataset(text_or_snippets, model=self.embed_model, 
                                  max_length=100, padding=True)
        else: raise TypeError("Unrecognized encoding mode")
        datalloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
        pbar = tqdm(enumerate(datalloader), total=len(datalloader), 
                    desc=f"enocding {mode}", disable=not(use_tqdm))
        
        all_embeds = []
        for step, batch in pbar:
            with torch.no_grad():
                enc_input_ids = batch.to(device)
                _,batch_embed = self.embed_model(enc_input_ids)
                for embed in batch_embed: all_embeds.append(embed)
                # if step == 5: break # DEBUG
        # print(type(all_embeds[0]), len(all_embeds))
        return all_embeds

    def fit(self, train_path: str, val_path: str, **args):
        exp_name = args.get("exp_name", "experiment")
        device_id = args.get("device_id", "cuda:0")
        batch_size = args.get("batch_size", 32)
        epochs = args.get("epochs", 5)
        
        use_AST = args.get("use_AST", False)
        sim_intents_path = args.get("sim_intents_path")
        perturbed_codes_path = args.get("perturbed_codes_path")
        intent_level_dynamic_sampling = args.get("intent_level_dynamic_sampling", False)
        
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
        
        config_path = os.path.join(exp_name, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f)
        print(f"saved config to {config_path}")
        print(f"model will be saved at {save_path}")
        print(f"moving model to {device}")
        self.embed_model.to(device)
        if intent_level_dynamic_sampling:
            from datautils import DynamicTriplesDataset
            
            assert sim_intents_path is not None, "Missing path to dictionary containing similar intents corresponding to an intent"
            sim_intents_map = json.load(open(sim_intents_path))
            perturbed_codes = {}
            if use_AST:
                assert perturbed_codes_path is not None, "Missing path to dictionary containing perturbed codes corresponding to a given code snippet"
                perturbed_codes = json.load(open(perturbed_codes_path))
            # creat the data loaders.
            trainset = DynamicTriplesDataset(
                train_path, "unixcoder",
                sim_intents_map=sim_intents_map,
                perturbed_codes=perturbed_codes,
                use_AST=use_AST, model=self, 
                device=device_id,
                max_length=100, padding=True,
            )
            valset = DynamicTriplesDataset(
                val_path, "unixcoder", model=self, 
                val=True, max_length=100, padding=True, 
            )
        else:
            trainset = TriplesDataset(train_path, model=self.embed_model, 
                                      max_length=100, padding=True)
            valset = TriplesDataset(val_path, model=self.embed_model, 
                                    max_length=100, padding=True)
        trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
        valloader = DataLoader(valset, shuffle=False, batch_size=batch_size)
        train_metrics = {
            "epochs": [],
            "summary": [],
        } 
        train_acc = TripletAccuracy()
        train_hard_neg_acc = TripletAccuracy()
        best_val_acc = 0
        for epoch_i in range(epochs):
            self.train()
            batch_losses = []
            pbar = tqdm(enumerate(trainloader), total=len(trainloader),
                        desc=f"train: epoch: {epoch_i+1}/{epochs} batch_loss: 0 loss: 0 acc: 0")
            train_acc.reset()
            train_hard_neg_acc.reset()
            for step, batch in pbar: 
                if args.get("dynamic_negative_sampling", False):
                    batch = dynamic_negative_sampling(
                        self.embed_model, batch, 
                        model_name="unixcoder", 
                        device=device, k=1
                    )
                self.train()
                anchor_title = batch[0].to(device)
                pos_snippet = batch[1].to(device)
                neg_snippet = batch[2].to(device)
                anchor_text_emb, pos_code_emb, neg_code_emb = self(anchor_title, pos_snippet, neg_snippet)
                batch_loss = self.loss_fn(anchor_text_emb, pos_code_emb, neg_code_emb)
                batch_loss.backward()
                self.optimizer.step()
                train_acc.update(anchor_text_emb, pos_code_emb, neg_code_emb)
                # scheduler.step()  # Update learning rate schedule
                self.zero_grad()
                batch_losses.append(batch_loss.item())
                MIX_STEP = ""
                HARD_ACC = ""
                if hasattr(trainset, "update"):
                    train_hard_neg_acc.update(
                        anchor_text_emb, pos_code_emb, 
                        neg_code_emb, batch[3].cpu(),
                    )
                    HARD_ACC = f" hacc: {100*train_hard_neg_acc.get():.2f}"
                    trainset.update(train_hard_neg_acc.get())
                    MIX_STEP = trainset.mix_step()
                pbar.set_description(f"train: epoch: {epoch_i+1}/{epochs} {MIX_STEP}batch_loss: {batch_loss:.3f} loss: {np.mean(batch_losses):.3f} acc: {100*train_acc.get():.2f}{HARD_ACC}")
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
    print("creating model object")
    triplet_net = UniXcoderTripletNet(**vars(args))
    print("commencing training")
    
    metrics = triplet_net.fit(exp_name=args.exp_name, epochs=args.epochs,
                              perturbed_codes_path=args.perturbed_codes_path,
                              device_id=args.device_id, val_path=args.val_path, 
                              train_path=args.train_path, batch_size=args.batch_size,
                              dynamic_negative_sampling=args.dynamic_negative_sampling,
                              sim_intents_path=args.sim_intents_path, use_AST=args.use_AST,
                              intent_level_dynamic_sampling=args.intent_level_dynamic_sampling)
    metrics_path = os.path.join(args.exp_name, "train_metrics.json")
    
    print(f"saving metrics to {metrics_path}")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

def test_retreival(args):
    print("initializing model ..")
    device = args.device_id if torch.cuda.is_available() else "cpu"
    
    ckpt_path = os.path.join(args.exp_name, "model.pt")
    print(f"loading checkpoint (state dict) from {ckpt_path}")
    try: state_dict = torch.load(ckpt_path, map_location="cpu")
    except Exception as e: 
        state_dict = None; print(e)
    
    print("creating model object")
    triplet_net = UniXcoderTripletNet(**vars(args))
    if state_dict: 
        print(f"loading state dict read from: \x1b[34;1m{ckpt_path}\x1b[0m")
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
                    "UniXcoder_zero_shot", 
                    f"test_metrics_{dist_func}_{setting}.json"
                )
                os.makedirs("UniXcoder_zero_shot", exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f)

                
if __name__ == "__main__":
    args = get_args()
    if args.train: main(args) # finetune.
    if args.test: test_retreival(args)
    if args.test_ood: 
        print("creating model object")
        # instantiate model class.
        triplet_net = UniXcoderTripletNet(**vars(args))
        test_ood_performance(
            triplet_net, model_name="unixcoder", args=args,
            query_paths=["query_and_candidates.json", "external_knowledge/queries.json", "data/queries_webquery.json"],
            cand_paths=["candidate_snippets.json", "external_knowledge/candidates.json", "data/candidates_webquery.json"], 
        )
    # setting in ['code', 'annot', 'code+annot']