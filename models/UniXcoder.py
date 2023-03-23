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
from sklearn.metrics import label_ranking_average_precision_score as MRR
from models.metrics import TripletAccuracy, recall_at_k, RuleWiseAccuracy
from models import test_ood_performance, dynamic_negative_sampling, fit_disco
from models.losses import scl_loss, TripletMarginWithDistanceLoss, cos_dist, cos_cdist, cos_csim

# set logging level of transformers.
import transformers
transformers.logging.set_verbosity_error()
# seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
VALID_STEPS = 501
SHUFFLE_BATCH_DEBUG_SETTING = True
# get arguments
def get_args():
    parser = argparse.ArgumentParser("script to train (using triplet margin loss), evaluate and predict with the UniXcoder in Late Fusion configuration for Neural Code Search.")
    parser.add_argument("-pp", "--predict_path", type=str, default="triples/triples_train.json", help="path to data for prediction of regression scores")
    parser.add_argument("-tp", "--train_path", type=str, default="triples/triples_train.json", help="path to training triplet data")
    parser.add_argument("-vp", "--val_path", type=str, default="triples/triples_test.json", help="path to validation triplet data")
    parser.add_argument("-c", "--candidates_path", type=str, default="candidate_snippets.json", help="path to candidates (to test retrieval)")
    parser.add_argument("-q", "--queries_path", type=str, default="query_and_candidates.json", help="path to queries (to test retrieval)")
    parser.add_argument("-en", "--exp_name", type=str, default="UniXcoder_rel_thresh", help="experiment name (will be used as folder name)")
    parser.add_argument("-w", "--warmup_steps", type=int, default=3000, help="no. of warmup steps (soft negatives only during warmup)")
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
    parser.add_argument("-p", "--p", type=int, default=2, help="the p used in mastering rate")
    parser.add_argument("-beta", "--beta", type=float, default=0.01, help="the beta used in the von-Mises fisher sampling")
    parser.add_argument("-nc", "--no_curriculum", action="store_true", help="turn of curriclum (only hard negatives)")
    parser.add_argument("-ccpp", "--code_code_pairs_path", type=str, default=None, 
                        help="path to code-code pairs for CodeRetriever's unimodal objective")
    parser.add_argument("-rc", "--rand_curriculum", action="store_true", help="random curriculum: equal probability of hard and soft negatives")
    parser.add_argument("-ast", "--use_AST", action="store_true", help="use AST perturbed negative samples")
    parser.add_argument("-idns", "--intent_level_dynamic_sampling", action="store_true", 
                        help="dynamic sampling based on similar intents")
    parser.add_argument("-crb", "--code_retriever_baseline", action="store_true", help="use CodeRetriever objective")
    parser.add_argument("-disco", "--disco_baseline", action="store_true", help="use DISCO training procedure")
    parser.add_argument("-uce", "--use_cross_entropy", action="store_true", help="use cross entropy loss instead of triplet margin loss")
    parser.add_argument("-ct", "--curr_type", type=str, default="mr", choices=['mr', 'rand', 'lp', 'exp', 'hard', "soft"],
                        help="""type of curriculum (listed below): 
                             1) mr: mastering rate based curriculum 
                             2) rand: equal prob. of hard & soft -ves
                             3) lp: learning progress based curriculum
                             4) exp: exponential decay with steps/epochs
                             5) hard: hard negatives only
                             6) soft: soft negatives only""")
    parser.add_argument("-igwr", "--ignore_worst_rules", action='store_true',
                        help="ignore the 6 worst/easiest perturbation rules")
    parser.add_argument("-discr", "--use_disco_rules", action='store_true',
                        help="use the rules outlined in/inspired by the DISCO paper (9)")
    parser.add_argument("-ccl", "--use_ccl", action="store_true", help="use code contrastive loss for hard negatives")
    parser.add_argument("-csim", "--use_csim", action="store_true", help="cosine similarity instead of euclidean distance")
    args = parser.parse_args()
    if args.use_cross_entropy and args.curr_type not in ["soft", "hard"]:
        args.curr_type = "hard"
    if args.use_ccl: args.curr_type = "hard"
    assert not(args.use_ccl and args.use_cross_entropy), "conflicting objectives selected: CCL and CE CL"
    assert not(args.use_ccl and args.code_retriever_baseline), "conflicting objectives selected: CCL and CodeRetriever"
    if args.code_retriever_baseline: # only use soft negative for CodeRetriever
        args.curr_type = "soft"

    return args # parser.add_argument("-cp", "--ckpt_path", type=str, default="UniXcoder_rel_thresh/model.pt")

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
    """To represent text streams"""
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

        
class ValRetDataset(Dataset):
    # JUST an engineering related class to convert NL-PL pairs to retrieval setting.
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
        intent_level_dynamic_sampling = args.get("intent_level_dynamic_sampling", False)
        self.config = {}
        self.config["model_path"] = model_path
        
        margin = args.get("margin", 1)
        dist_fn_deg = args.get("dist_fn_deg", 2)
        # store config info.
        self.ignore_worst_rules = args.get("ignore_worst_rules", False)
        self.ignore_non_disco_rules = args.get("use_disco_rules", False)
        self.code_retriever_baseline = args.get("code_retriever_baseline", False)
        self.use_cross_entropy = args.get("use_cross_entropy", False)
        self.use_ccl = args.get("use_ccl", False)
        self.use_scl = args.get("use_scl", False)
        self.use_csim = args.get("use_csim", False)
        
        self.config["margin"] = margin
        self.config["dist_fn_deg"] = dist_fn_deg
        print(f"loading pretrained UniXcoder embedding model from {model_path}")
        start = time.time()
        self.embed_model = UniXcoder(model_path, tok_path="~/unixcoder-base-tok")
        self.tokenizer = self.embed_model.tokenize
        print(f"loaded embedding model in {(time.time()-start):.2f}s")
        # print(f"loaded tokenizer files from {tok_path}")
        # optimizer and loss.
        adam_eps = 1e-8
        lr = args.get("lr", 1e-5)
        self.config["lr"] = lr
        print(f"optimizer = AdamW(lr={lr}, eps={adam_eps})")
        self.optimizer = AdamW(self.parameters(), eps=adam_eps, lr=lr)
        
        if intent_level_dynamic_sampling:
            self.hard_neg_loss_fn = nn.TripletMarginLoss(margin=1, p=dist_fn_deg, reduction="none")
            self.soft_neg_loss_fn = nn.TripletMarginLoss(margin=margin, p=dist_fn_deg, reduction="none")
            print(f"hard_neg_loss_fn = TripletMarginLoss(margin=1, p={dist_fn_deg})")
            print(f"soft_neg_loss_fn = TripletMarginLoss(margin={margin}, p={dist_fn_deg})")
            self.config["hard_neg_loss_fn"] = f"{self.hard_neg_loss_fn}"
            self.config["soft_neg_loss_fn"] = f"{self.soft_neg_loss_fn}"
            
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=dist_fn_deg)
        print(f"loss_fn = TripletMarginLoss(margin={margin}, p={dist_fn_deg})")
        self.config["loss_fn"] = f"{self.loss_fn}"
        self.config["optimizer"] = f"{self.optimizer}"
        self.config["ignore_worst_rules"] = self.ignore_worst_rules
        self.config["use_disco_rules"] = self.ignore_non_disco_rules
        self.config["code_retriever_baseline"] = self.code_retriever_baseline
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, anchor_title, pos_snippet, neg_snippet):
        _,anchor_text_emb = self.embed_model(anchor_title)
        _,pos_code_emb = self.embed_model(pos_snippet)
        _,neg_code_emb = self.embed_model(neg_snippet)
        
        return anchor_text_emb, pos_code_emb, neg_code_emb
        
    def val(self, valloader: DataLoader, epoch_i: int=0, 
            epochs: int=0, device="cuda:0"):
        self.eval()
        val_acc = TripletAccuracy(margin=0)
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
    
    def val_ret(self, valset: Dataset, device="cuda:0"):
        self.eval()
        # get queries and candidates from validation set and encode them.
        labels = valset.get_labels()
        queries = valset.get_queries()
        candidates = valset.get_candidates()
        print(f"encoding {len(queries)} queries:")
        query_mat = self.encode_emb(queries, mode="text", batch_size=48,
                                    use_tqdm=True, device_id=device)
        query_mat = torch.stack(query_mat)
        print(f"encoding {len(candidates)} candidates:")
        cand_mat = self.encode_emb(candidates, mode="code", batch_size=48,
                                   use_tqdm=True, device_id=device)
        # score and rank documents.
        cand_mat = torch.stack(cand_mat)
        scores = torch.cdist(query_mat, cand_mat, p=2)
        doc_ranks = scores.argsort(axis=1)
        recall_at_5 = recall_at_k(labels, doc_ranks.tolist(), k=5)
        
        return recall_at_5
        
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
        warmup_steps = args.get("warmup_steps", 3000)
        exp_name = args.get("exp_name", "experiment")
        device_id = args.get("device_id", "cuda:0")
        batch_size = args.get("batch_size", 32)
        epochs = args.get("epochs", 5)
        beta = args.get("beta", 0.01)
        p = args.get("p", 2)
        curriculum_type = args.get("curriculum_type")
        # use_curriculum = not(args.get("no_curriculum", False))
        # rand_curriculum = args.get("rand_curriculum", False)
        use_AST = args.get("use_AST", False)
        sim_intents_path = args.get("sim_intents_path")
        code_code_pairs_path = args.get("code_code_pairs_path")
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
        self.config["use_AST"] = use_AST
        self.config["intent_level_dynamic_sampling"] = intent_level_dynamic_sampling
        
        print(f"model will be saved at {save_path}")
        print(f"moving model to {device}")
        self.embed_model.to(device)
        if intent_level_dynamic_sampling or use_AST:
            from datautils import DynamicTriplesDataset
            perturbed_codes = {}
            sim_intents_map = {}
            
            if intent_level_dynamic_sampling:
                assert sim_intents_path is not None, "Missing path to dictionary containing similar intents corresponding to an intent"
                sim_intents_map = json.load(open(sim_intents_path))
            
            if use_AST:
                assert perturbed_codes_path is not None, "Missing path to dictionary containing perturbed codes corresponding to a given code snippet"
                perturbed_codes = json.load(open(perturbed_codes_path))
            # creat the data loaders.
            # trainset = DynamicTriplesDataset(
            #     train_path, "unixcoder", device=device_id, beta=beta, warmup_steps=warmup_steps,
            #     sim_intents_map=sim_intents_map, perturbed_codes=perturbed_codes,
            #     use_curriculum=use_curriculum, rand_curriculum=rand_curriculum,
            #     use_AST=use_AST, model=self, p=p, max_length=100, padding=True,
            # )
            trainset = DynamicTriplesDataset(
                train_path, "unixcoder", device=device_id, beta=beta, p=p, warmup_steps=warmup_steps,
                use_AST=use_AST, model=self, tokenizer=self.tokenizer, sim_intents_map=sim_intents_map, 
                perturbed_codes=perturbed_codes, curriculum_type=curriculum_type,                 
                # use_curriculum=use_curriculum, rand_curriculum=rand_curriculum,
                ignore_non_disco_rules=self.ignore_non_disco_rules,
                max_length=100, padding=True,
            )
            valset = ValRetDataset(val_path)
            # valset = DynamicTriplesDataset(
            #     val_path, "unixcoder", model=self, 
            #     val=True, max_length=100, padding=True, 
            # )
            self.config["trainset.warmup_steps"] = trainset.warmup_steps # no. of warmup steps before commencing training.
            self.config["trainset.epsilon"] = trainset.epsilon # related to mastering rate.
            self.config["trainset.delta"] = trainset.soft_master_rate.delta # related to mastering rate.
            self.config["trainset.beta"] = trainset.beta # related to hard negative sampling.
            self.config["trainset.p"] = trainset.soft_master_rate.p # related to mastering rate.
        elif self.code_retriever_baseline:
            from datautils import CodeRetrieverDataset
            trainset = CodeRetrieverDataset(
                train_path, code_code_path=code_code_pairs_path, model_name="unixcoder", 
                tokenizer=self.tokenizer, max_length=100, padding=True,
                # max_length=100, padding="max_length", return_tensors="pt", add_special_tokens=True, truncation=True,
            )
            valset = ValRetDataset(val_path)
        else:
            trainset = TriplesDataset(train_path, model=self.embed_model, 
                                      max_length=100, padding=True)
            valset = TriplesDataset(val_path, model=self.embed_model, 
                                    max_length=100, padding=True)
        config_path = os.path.join(exp_name, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f)
        print(f"saved config to {config_path}")
        if SHUFFLE_BATCH_DEBUG_SETTING and not(self.code_retriever_baseline): #TODO: remove this. Used only for a temporary experiment.
            from datautils import batch_shuffle_collate_fn
            # trainloader = DynamicDataLoader(trainset, shuffle=True, batch_size=batch_size, 
            #                                 model=self.embed_model, device=device)
            trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size,
                                     collate_fn=batch_shuffle_collate_fn)
            valloader = DataLoader(valset, shuffle=False, batch_size=batch_size,
                                   collate_fn=batch_shuffle_collate_fn)
        else:
            trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
            valloader = DataLoader(valset, shuffle=False, batch_size=batch_size)
        train_metrics = {
            "log_steps": [],
            "summary": [],
        } 
        rule_wise_acc = RuleWiseAccuracy(margin=1, use_scl=self.use_scl)
        if not(self.use_cross_entropy or self.code_retriever_baseline):
            train_soft_neg_acc = TripletAccuracy(margin=1, use_scl=self.use_scl)
            train_hard_neg_acc = TripletAccuracy(margin=1, use_scl=self.use_scl)
        else: 
            train_tot = 0
            train_acc = 0
            train_u_acc = 0
        best_val_acc = 0
        for epoch_i in range(epochs):
            self.train()
            batch_losses = []
            pbar = tqdm(enumerate(trainloader), total=len(trainloader),
                        desc=f"train: epoch: {epoch_i+1}/{epochs} batch_loss: 0 loss: 0 acc: 0")
            rule_wise_acc.reset()
            if not(self.use_cross_entropy or self.code_retriever_baseline):
                train_soft_neg_acc.reset()
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
                N = len(batch[0])
                # if intent_level_dynamic_sampling:
                #     b = batch[3].to(device)
                #     embs = (anchor_text_emb, pos_code_emb, neg_code_emb)
                #     batch_loss = (b.float()*self.hard_neg_loss_fn(*embs) + (~b).float()*self.soft_neg_loss_fn(*embs)).mean()
                # else: batch_loss = self.loss_fn(anchor_text_emb, pos_code_emb, neg_code_emb)
                if hasattr(trainset, "update") or isinstance(trainset, CodeRetrieverDataset):
                    if not(self.use_cross_entropy or self.code_retriever_baseline):
                        train_soft_neg_acc.update(
                            anchor_text_emb, pos_code_emb, 
                            neg_code_emb, (batch[-1]==0).cpu(),
                        )
                        train_hard_neg_acc.update(
                            anchor_text_emb, pos_code_emb, 
                            neg_code_emb, (batch[-1]!=0).cpu(),
                        )
                        trainset.update(
                            train_soft_neg_acc.last_batch_acc,
                            train_hard_neg_acc.last_batch_acc,
                        )
                        HARD_ACC = f" ha:{100*train_hard_neg_acc.get():.2f}"
                        MIX_STEP = trainset.mix_step()
                    if self.use_scl:
                        batch_loss = scl_loss(
                            anchor_text_emb, pos_code_emb, 
                            neg_code_emb, lamb=1, device=device,
                            loss_fn=self.loss_fn,
                        ).mean()
                        pd_ap = F.pairwise_distance(anchor_text_emb, pos_code_emb).mean().item()
                        pd_an = F.pairwise_distance(anchor_text_emb, neg_code_emb).mean().item()
                        pd_ap_an_info = f" ap:{pd_ap:.3f} an:{pd_an:.3f}"
                        # hard_loss = self.loss_fn(anchor_text_emb, torch.zeros_like(
                        #                          pos_code_emb), neg_code_emb)
                        # soft_loss = self.loss_fn(anchor_text_emb, pos_code_emb, neg_code_emb)
                        # batch[-1] = batch[-1].to(device)
                        # batch_loss = (batch[-1]*hard_loss + (~batch[-1])*soft_loss).mean()
                    elif self.use_cross_entropy:
                        d_ap = torch.cdist(anchor_text_emb, pos_code_emb)
                        d_an = torch.cdist(anchor_text_emb, neg_code_emb)
                        scores = -torch.cat((d_ap, d_an), axis=-1)
                        target = torch.as_tensor(range(N)).to(device)
                        batch_loss = self.ce_loss(scores, target)
                        preds = scores.argmax(dim=-1)
                        train_acc += (preds == target).sum().item()
                        train_tot += N
                        batch_loss_str = f"bl:{batch_loss:.3f}"
                        metric_str = f"a:{(100*train_acc/train_tot):.2f}"
                    elif self.code_retriever_baseline:
                        if self.use_csim:
                            d_ap = -cos_csim(anchor_text_emb, pos_code_emb)
                            d_pn = -cos_csim(pos_code_emb, neg_code_emb)
                        else:
                            d_ap = torch.cdist(anchor_text_emb, pos_code_emb)
                            d_pn = torch.cdist(pos_code_emb, neg_code_emb)
                        # margin = self.config['margin']*torch.eye(N).to(device)
                        target = torch.as_tensor(range(N)).to(device)
                        unimodal_loss = self.ce_loss(-d_ap, target)
                        bimodal_loss = self.ce_loss(-d_pn, target)
                        # unimodal_loss = self.ce_loss(-(d_ap+margin), target)
                        # bimodal_loss = self.ce_loss(-(d_pn+margin), target)
                        batch_loss = unimodal_loss + bimodal_loss
                        b_preds = (-d_ap).argmax(dim=-1)
                        u_preds = (-d_pn).argmax(dim=-1)
                        train_acc += (b_preds == target).sum().item()
                        train_u_acc += (u_preds == target).sum().item()
                        train_tot += N
                        metric_str = f"ba:{(100*train_acc/train_tot):.2f} ua:{(100*train_u_acc/train_tot):.2f}"
                        batch_loss_str = f"bl:{batch_loss:.3f}={unimodal_loss:.3f}u+{bimodal_loss:.3f}b"
                    else:
                        # pd_ap = F.pairwise_distance(anchor_text_emb, pos_code_emb)
                        # pd_an = F.pairwise_distance(anchor_text_emb, neg_code_emb)
                        # hard_neg_ctr = (pd_ap > pd_an).sum().item()
                        # pd_ap_an_info = f" ap:{pd_ap.mean().item():.3f} an:{pd_an.mean().item():.3f} {hard_neg_ctr}/{batch_size}"
                        if self.use_ccl: # use code contrastive loss (by default all negatives are hard negatives)
                            """the self distance (diagonal terms) in d_pp will always be zero
                            the cross distance is always positive so a code is always more similar to itself
                            than other codes. To overcome this we can add a margin term (a diagonal matrix) 
                            to d_pp to make sure the pos_code_emb has at least distance equal to this margin
                            compared to any other negative. Here we take this margin to be the same as the 
                            margin for the triplet margin loss."""
                            # margin = self.config["margin"]*torch.eye(N).to(device)
                            S_pp = cos_csim(self.dropout1(pos_code_emb), self.dropout2(pos_code_emb))
                            S_pn = cos_csim(self.dropout1(pos_code_emb), neg_code_emb)
                            # scores = -torch.cat((d_pp+margin, d_pn), axis=-1)
                            scores = torch.cat((S_pp, S_pn), axis=-1)
                            target = torch.as_tensor(range(N)).to(device)
                            soft_margin_loss = self.loss_fn(anchor_text_emb, pos_code_emb, 
                                                            pos_code_emb[torch.randperm(N)]).mean()
                            # hard_margin_loss = self.loss_fn(anchor_text_emb, pos_code_emb, 
                            #                                 neg_code_emb).mean()
                            ccl_loss = self.ce_loss(scores, target)
                            batch_loss = soft_margin_loss + ccl_loss
                            batch_loss_str = f"bl:{batch_loss:.3f}={soft_margin_loss:.3f}+{ccl_loss:.3f}"
                            # batch_loss = soft_margin_loss + hard_margin_loss + ccl_loss
                            # batch_loss_str = f"bl:{batch_loss:.3f}={soft_margin_loss:.3f}+{hard_margin_loss:.3f}+{ccl_loss:.3f}"
                        else: 
                            batch_loss = self.loss_fn(anchor_text_emb, pos_code_emb, neg_code_emb).mean()
                            batch_loss_str = f"bl:{batch_loss:.3f}"
                    rule_wise_acc.update(anchor_text_emb, pos_code_emb, 
                                         neg_code_emb, batch[-1].cpu().tolist())
                    if (self.use_cross_entropy or self.code_retriever_baseline):
                        pbar.set_description(f"T e:{epoch_i+1}/{epochs} bl:{batch_loss:.3f} l:{np.mean(batch_losses):.3f} {metric_str}")
                    else: 
                        pbar.set_description(
                            f"T e:{epoch_i+1}/{epochs} {MIX_STEP}{batch_loss_str} l:{np.mean(batch_losses):.3f} a:{100*train_soft_neg_acc.get():.2f}{HARD_ACC}"
                        )
                else: 
                    train_soft_neg_acc.update(
                        anchor_text_emb, 
                        pos_code_emb, neg_code_emb
                    )
                    batch_loss = self.loss_fn(anchor_text_emb, pos_code_emb, neg_code_emb).mean()
                    pbar.set_description(f"train: epoch: {epoch_i+1}/{epochs} batch_loss: {batch_loss:.3f} loss: {np.mean(batch_losses):.3f} acc: {100*train_soft_neg_acc.get():.2f}")
                batch_loss.backward()
                self.optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                self.zero_grad()
                batch_losses.append(batch_loss.item())
                # if step == 5: break # DEBUG
                if ((step+1) % VALID_STEPS == 0) or ((step+1) == len(trainloader)):
                    # validate current model
                    print(rule_wise_acc())
                    print(dict(rule_wise_acc.counts))
                    if intent_level_dynamic_sampling or use_AST or self.code_retriever_baseline:
                        # here the val_acc is actually recall@1 per batch, and averaged over mini-batches.
                        s = time.time()
                        val_acc = self.val_ret(valset, device=device)
                        print(f"validated in {time.time()-s}s")
                        print(f"recall@5 = {100*val_acc:.3f}")
                        val_loss = None
                    else:
                        val_acc, val_loss = self.val(valloader, epoch_i=epoch_i, 
                                                     epochs=epochs, device=device)
                    # save model only after warmup is complete.
                    if val_acc > best_val_acc and (not(hasattr(trainset, "warmup_steps")) or trainset.warmup_steps == 0):
                        print(f"saving best model till now with val_acc: {val_acc} at {save_path}")
                        best_val_acc = val_acc
                        torch.save(self.state_dict(), save_path)

                    train_metrics["log_steps"].append({
                        "train_batch_losses": batch_losses, 
                        "train_loss": np.mean(batch_losses), 
                        "val_loss": val_loss,
                        "val_acc": 100*val_acc,
                    })
                    if (self.use_cross_entropy or self.code_retriever_baseline):
                        train_metrics["train_acc"] = 100*train_acc/train_tot
                        if self.code_retriever_baseline:
                            train_metrics["train_u_acc"] = 100*train_u_acc/train_tot
                    else:
                        train_metrics["train_soft_neg_acc"] = 100*train_soft_neg_acc.get()
                        train_metrics["train_hard_neg_acc"] = 100*train_hard_neg_acc.get()
                    metrics_path = os.path.join(exp_name, "train_metrics.json")
                    print(f"saving metrics to {metrics_path}")
                    with open(metrics_path, "w") as f:
                        json.dump(train_metrics, f)
            if self.code_retriever_baseline: trainset.reset()
        
        return train_metrics
    
def main(args):
    print("creating model object")
    triplet_net = UniXcoderTripletNet(**vars(args))
    print("commencing training")
    if args.disco_baseline: 
        metrics = fit_disco(triplet_net, model_name="unixcoder", **vars(args))
    else:
        metrics = triplet_net.fit(exp_name=args.exp_name, epochs=args.epochs,
                                  perturbed_codes_path=args.perturbed_codes_path,
                                  device_id=args.device_id, val_path=args.val_path, 
                                  train_path=args.train_path, batch_size=args.batch_size,
                                  beta=args.beta, p=args.p, warmup_steps=args.warmup_steps,
                                  dynamic_negative_sampling=args.dynamic_negative_sampling,
                                  sim_intents_path=args.sim_intents_path, use_AST=args.use_AST,
                                  intent_level_dynamic_sampling=args.intent_level_dynamic_sampling,
                                  no_curriculum=args.no_curriculum, rand_curriculum=args.rand_curriculum,
                                  code_code_pairs_path=args.code_code_pairs_path, curriculum_type=args.curr_type)
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