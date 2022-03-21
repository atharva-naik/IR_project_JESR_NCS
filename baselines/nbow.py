#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Atharva Naik (18CS10067)
# Neural bag of words baseline for text - code and text - text retrieval.
import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Tuple, Union, List
from transformers import RobertaModel, RobertaTokenizer


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