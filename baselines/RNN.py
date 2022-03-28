#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Atharva Naik (18CS10067)
# RNN baseline for text - code and text - text retrieval.
import torch
import torch.nn as nn

def random_masks(batch_size: int, padded_seq_len=100, min_seq_len=30) -> torch.Tensor:
    import random
    sampled_masks = []
    for i in range(batch_size):
        k = random.randint(min_seq_len, padded_seq_len)
        sampled_masks.append([1 for i in range(k)]+[0 for i in range(padded_seq_len-k)])
    
    return torch.as_tensor(sampled_masks)

class RNNEncoder(nn.Module):
    def __init__(self, embedding=None, **args):
        super(RNNEncoder, self).__init__()
        # get arguments.
        # dropout rate for not keeping.
        dropout = args.get("dropout", 0.2)
        self.device = args.get("device", "cpu")
        # number of layers in stacked LSTM.
        num_layers = args.get("num_layers", 2)
        self.num_layers = num_layers
        # sequence length (padded to this length.)
        seq_len = args.get("seq_len", 100)
        hidden_size = args.get("hidden_size", 64)
        self.hidden_size = hidden_size
        # this is determined by the tokenizer.
        embed_size = args.get("embed_size", 128)
        vocab_size = args.get("vocab_size", 50265)
        # if LSTM is bi-directional or not.
        is_bidirectional = args.get("is_bidirectional", True)
        if is_bidirectional:
            out_size = 2*hidden_size
            self.effective_num_layers = 2*self.num_layers
        else: 
            out_size = hidden_size
            self.effective_num_layers = self.num_layers
        self.out_size = out_size
        # weighted linear layer and it's activation.
        self.weighted_sum_linear = nn.Linear(out_size, 1)
        self.softmax = nn.Softmax(dim=-1)
        # # dropout layer.
        # self.dropout = nn.Dropout(dropout)
        # create the embedding layer.
        if embedding is not None:
            self.embed = embedding
            embed_size = embedding.embedding_dim
        else:
            self.embed = nn.Embedding(vocab_size, 
                                      embed_size, 
                                      padding_idx=1)
        # rnn encoder model.
        self.rnn = nn.LSTM(
            dropout=dropout,
            batch_first=True,
            num_layers=num_layers,
            input_size=embed_size,
            hidden_size=hidden_size,
            bidirectional=is_bidirectional,
        ) 
        
    def forward(self, input_ids, attn_masks):
        batch_size = input_ids.shape[0]
        input_embeds = self.embed(input_ids)
        # print("input_embeds", input_embeds.shape)
        # intialize hidden and cell states.
        h0 = torch.randn(self.effective_num_layers, 
                         batch_size, self.hidden_size).to(self.device)
        c0 = torch.randn(self.effective_num_layers, 
                         batch_size, self.hidden_size).to(self.device)
        # print("h0:", h0.shape)
        # print("c0:", c0.shape)
        # get output, final hidden state and finall cell state.
        output, (hn, cn) = self.rnn(input_embeds, (h0, c0))
        # output shape -> (batch_size, seq_len, 2*hidden_size) (the 2 is because of bi-directionality)
        return self.pool_sequence_embedding(
            "weighted_mean", 
            output, attn_masks
        )

    def pool_sequence_embedding(self, pool_mode: str,
                                seq_token_embeds: torch.Tensor,
                                seq_token_masks: torch.Tensor) -> torch.Tensor:
        """
        Takes a batch of sequences of token embeddings and applies a pooling function,
        returning one representation for each sequence.
        Args:
            pool_mode: The pooling mode, one of "mean", "max", "weighted_mean". For
             the latter, a weight network is introduced that computes a score (from [0,1])
             for each token, and embeddings are weighted by that score when computing
             the mean.
            sequence_token_embeddings: A float32 tensor of shape [B, T, D], where B is the
             batch dimension, T is the maximal number of tokens per sequence, and D is
             the embedding size.
            sequence_lengths: An int32 tensor of shape [B].
            sequence_token_masks: A float32 tensor of shape [B, T] with 0/1 values used
             for masking out unused entries in sequence_embeddings.
        Returns:
            A tensor of shape [B, D], containing the pooled representation for each
            sequence.
        """
        D = seq_token_embeds.shape[-1]
        # batch_size x seq_len -> batch_size x seq_len x embed_dim
        seq_token_masks = seq_token_masks.unsqueeze(dim=-1).repeat(1,1,D) 
        if pool_mode == 'mean':
            return (seq_token_embeds * seq_token_masks).sum(1)/seq_token_masks.sum(1)
        elif pool_mode == 'weighted_mean':
            # weighted_sum_linear gives you batch_size x seq_len x 1
            # softmax just normalizes the token weights
            # if you don't squeeze the output will not properly sum to 1.
            token_weights = self.softmax(
                self.weighted_sum_linear(seq_token_embeds).squeeze()
            )
            # reshape the token_weights for multiplication purposes.
            token_weights = token_weights.unsqueeze(dim=-1).repeat(1, 1, self.out_size)
            # apply the masks.
            token_weights *= seq_token_masks
            return (seq_token_embeds * token_weights).sum(1) / token_weights.sum(1) 
        else:
            raise ValueError("Unknown sequence pool mode '%s'!" % pool_mode)