#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from typing import Tuple
from transformers.optimization import AdamW
from torch.datautils import Dataset, DataLoader


class NCSPipeLine(nn.Module):
    def __init__(self, text_encoder: nn.Module, 
                 code_encoder: nn.Module, 
                 sim_model: nn.Module, **agrs):
        """
        code to train, test and predict using the Joint Space Embedding Retrieval NCS model.
        """
        super(NCSPipeLine, self).__init__()
        self.text_encoder = text_encoder
        self.code_encoder = code_encoder
        self.sim_model = sim_model
        self.loss_fn = nn.TripletMarginLoss()
        self.optimizer = nn.AdamW(eps=1e-8)
        
    def forward(self, text_enc_args: Union[
                Tuple[torch.Tensor], Tuple[torch.Tensor, Torch.Tensor], 
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], 
                code_enc_args: Union[
                Tuple[torch.Tensor], Tuple[torch.Tensor, Torch.Tensor], 
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        with torch.no_grad():
            text_emb = self.text_encoder(*text_enc_args)
            code_emb = self.code_encoder(*code_enc_args)
        score = self.sim_model(text_emb, code_emb)
        
        return score
    
    def train(self, train_loader: DataLoader, 
              val_loader: DataLoader, **args):
        # training args.
        lr = args.get("lr", 1e-5)
        epochs = args.get("epochs", 20)
        device = torch.cuda(args.get("device_id"), 0)
        # move model to device
        self.to(device)
        # train metrics JSON
        train_metrics = {
            "epochs": []
            "summary": []
        } 
        # main train-eval loop
        for epoch_i in range(epochs):
            batch_losses = []
            # train step.
            for step, batch in enumerate(train_loader):
                batch[0] = batch[0].to(device)
                batch[1] = batch[1].to(device)
                scores = self(batch[0], batch[1])
                labels = batch[2]
                scores = scores.to("cpu")
                bath_loss = self.loss_fn(scores, labels)
                # clear previous gradients.
                self.optimizer.zero_grad()
                # calculate gradients and optimize.
                batch_loss.backward()
                self.optimizer.step()
                batch_losses.append(batch_loss)
            # validation step.
            # for step, batch in enumerate(val_loader):
            #     with torch.no_grad():
            train_metrics["epochs"].append({"batch_losses": batch_losses})
        
        return train_metrics
                    