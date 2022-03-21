import torch
import torch.nn as nn


class RNNEncoder(nn.Module):
    def __init__(self):
        super(RNNEncoder, self).__init__()
        