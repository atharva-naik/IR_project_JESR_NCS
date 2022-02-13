#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from model.pipeline import NCSPipeLine 
from model.encoder.code import RobertaCode
from model.encoder.text import RobertaEncoder

# Initialize the NCS pipeline
def init_model():
    text_encoder = RobertaEncoder("~/text", path_type="local")
    code_encoder = RoberaCode
    
    