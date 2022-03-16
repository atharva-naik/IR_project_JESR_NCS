#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from transformers import RobertaModel

# Roberta text encoder.
class RobertaEncoder(nn.Module):
    def __init__(self, ,  **args):
        self.model = RobertaModel.from_pretrained()