#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Atharva Naik (18CS10067)
# CNN baseline for text - code and text - text retrieval.
import os
import json
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from typing import Tuple, Union, List
from transformers import RobertaModel, RobertaTokenizer

def random_masks(batch_size: int, padded_seq_len=100, min_seq_len=30) -> torch.Tensor:
    import random
    sampled_masks = []
    for i in range(batch_size):
        k = random.randint(min_seq_len, padded_seq_len)
        sampled_masks.append([1 for i in range(k)]+[0 for i in range(padded_seq_len-k)])
    
    return torch.as_tensor(sampled_masks)

class CNNEncoder(nn.Module):
    def __init__(self, embedding=None, **args):
        super(CNNEncoder, self).__init__()
        dropout = args.get("dropout", 0.2)
        seq_len = args.get("seq_len", 100)
        num_filters = args.get("num_filters", [128, 128, 128])
        kernel_sizes = args.get("kernel_sizes", [16, 16, 16])
        vocab_size = args.get("vocab_size", 50265)
        embed_size = args.get("embed_size", 128)
        self.num_filters = num_filters
        # weighted linear layer and it's activation.
        self.weighted_sum_linear = nn.Linear(num_filters[-1], 1)
        self.softmax = nn.Softmax(dim=-1)
        # dropout layer.
        self.dropout = nn.Dropout(dropout)
        # create the embedding layer.
        if embedding is not None:
            self.embed = embedding
            embed_size = embedding.embedding_dim
        else:
            self.embed = nn.Embedding(vocab_size, 
                                      embed_size, 
                                      padding_idx=1)
        kernel_widths_and_num_filters = zip(kernel_sizes[1:], num_filters[1:])
        # for stride=1, no dilation. we have the same padding amount across all layers as 
        # they will have input of same size after the padding and the `num_filters` and 
        # `kerne_size` are identical.
        self.padding_amt = (seq_len - (seq_len - kernel_sizes[0] + 1))
        # creat the convolution layers.
        conv_layers = [nn.Conv1d(
                in_channels=embed_size, 
                out_channels=num_filters[0], 
                kernel_size=kernel_sizes[0],
            )
        ]
        in_channels = num_filters[0]
        for kernel_width, out_channels in kernel_widths_and_num_filters:
            conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=kernel_width,
                )
            )
            in_channels = out_channels
        self.convs = nn.ModuleList(conv_layers)
        # activation function
        self.act_fn = nn.Tanh()
        
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
        elif pool_mode == 'max':
            sequence_token_masks = -BIG_NUMBER * (1 - sequence_token_masks)  # B x T
            sequence_token_masks = tf.expand_dims(sequence_token_masks, axis=-1)  # B x T x 1
            return tf.reduce_max(sequence_token_embeddings + sequence_token_masks, axis=1)
        elif pool_mode == 'weighted_mean':
            # weighted_sum_linear gives you batch_size x seq_len x 1
            # softmax just normalizes the token weights
            # if you don't squeeze the output will not properly sum to 1.
            token_weights = self.softmax(
                self.weighted_sum_linear(seq_token_embeds).squeeze()
            )
            # reshape the token_weights for multiplication purposes.
            token_weights = token_weights.unsqueeze(dim=-1).repeat(1, 1, self.num_filters[-1])
            # apply the masks.
            token_weights *= seq_token_masks
            return (seq_token_embeds * token_weights).sum(1) / token_weights.sum(1) 
    #         token_weights = tf.layers.dense(sequence_token_embeddings,
    #                                         units=1,
    #                                         activation=tf.sigmoid,
    #                                         use_bias=False)  # B x T x 1
    #         token_weights *= tf.expand_dims(sequence_token_masks, axis=-1)  # B x T x 1
    #         seq_embedding_weighted_sum = tf.reduce_sum(sequence_token_embeddings * token_weights, axis=1)  # B x D
    #         return seq_embedding_weighted_sum / (tf.reduce_sum(token_weights, axis=1) + 1e-8)  # B x D
        else:
            raise ValueError("Unknown sequence pool mode '%s'!" % pool_mode)
        
    def forward(self, input_ids, attention_masks):
        enc_input = self.embed(input_ids)
        curr_enc = enc_input.transpose(-2,-1) # (batch_size, seq_len, emb_size) (e.g. (32, 100, 128) -> (batch_size, emb_size, seq_len) (e.g. 32, 128, 100))
        # the `enc_input` has the format NWC, but Conv1d expects NCW, so we need to reshape `enc_input`
        for i in range(len(self.convs)):
            next_enc = self.convs[i](curr_enc)
            # print("next_enc:", next_enc.shape)
            # apply padding explicitly as padding=`same` is not available before pytorch 1.9
            next_enc = F.pad(next_enc, (0, self.padding_amt))
            # print("next_enc(after pad):", next_enc.shape)
            # residual connection (requires padding of output to work).
            # the if condition below is to handle the case of codebert init.
            if next_enc.shape[1] == curr_enc.shape[1]:
                next_enc = next_enc + curr_enc
            # apply the activation,
            curr_enc = self.act_fn(next_enc)
            # apply dropout.
            curr_enc = self.dropout(curr_enc)
        # transpose back to original shape.
        curr_enc = curr_enc.transpose(-2,-1)
        
        return self.pool_sequence_embedding(
            "weighted_mean", curr_enc, 
            attention_masks
        )

# class CNNEncoder(nn.Module):
#     def __init__(self, **args):
#         super(CNNEncoder, self).__init__()
#         dropout = args.get("dropout", 0.2)
#         seq_len = args.get("seq_len", 100)
#         kernel_sizes = args.get("kernel_sizes", [3, 4, 5])
#         vocab_size = args.get("vocab_size", 50265)
#         static = False
        
#         V = vocab_size
#         D = 128 # args.embed_dim
#         # C = args.class_num
#         Ci = 1
#         Co = len(kernel_sizes)
#         Ks = kernel_sizes

#         self.embed = nn.Embedding(V, D, padding_idx=1)
#         self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
#         self.dropout = nn.Dropout(dropout)
#         # self.fc1 = nn.Linear(len(Ks) * Co, C)
#         if static:
#             self.embed.weight.requires_grad = False

#     def forward(self, x):
#         x = self.embed(x)  # (N, W, D)
    
#         x = x.unsqueeze(1)  # (N, Ci, W, D)

#         x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

#         x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

#         x = torch.cat(x, 1)

#         x = self.dropout(x)  # (N, len(Ks)*Co)
#         # logit = self.fc1(x)  # (N, C)
#         return logit


# class CNNEncoder(nn.Module):
#     def __init__(self, embedding=None, **args):
#         super(CNNEncoder, self).__init__()
#         # Parameters for CNNEncoder.
#         dropout = args.get("dropout", 0.2)
#         seq_len = args.get("seq_len", 100)
#         kernel_sizes = args.get("kernel_sizes", [3, 4, 5])
#         self.kernel_sizes = kernel_sizes
#         # Dropout definition
#         self.dropout = nn.Dropout(dropout)
#         # Output size for each convolution
#         self.out_size = args.get("out_size", 128)
#         # Number of strides for each convolution
#         self.stride = args.get("stride", 1)
#         # Embedding layer definition (we just pass codebert's embedding layer here).
#         if embedding is None:
#             vocab_size = args.get("vocab_size", 50265)
#             padding_idx = args.get("padding_idx", 1)
#             self.embedding_size = args.get("embedding_size", 128)
#             self.embedding = nn.Embedding(vocab_size, 
#                                           self.embedding_size, 
#                                           padding_idx=padding_idx)
#         else:
#             self.embedding = embedding  
#             self.embedding_size = embedding.embedding_dim
#         # self.layer_norm = nn.LayerNorm([self.embedding_size])
#         # self.tanh = nn.Tanh()
#         # the output embedding size can be edited, but default to size of codebert's word embeddings (768)
#         output_embed_size = args.get("output_embed_size", self.embedding_size)
#         # initialize layers
#         # Convolution layers definition
#         self.conv_0 = nn.Conv1d(seq_len, self.out_size, kernel_sizes[0], self.stride)
#         self.conv_1 = nn.Conv1d(seq_len, self.out_size, kernel_sizes[1], self.stride)
#         self.conv_2 = nn.Conv1d(seq_len, self.out_size, kernel_sizes[2], self.stride)
#         self.num_conv_layers = 3
#         # Pooling layers definition
#         self.pool_0 = nn.MaxPool1d(kernel_sizes[0], self.stride)
#         self.pool_1 = nn.MaxPool1d(kernel_sizes[1], self.stride)
#         self.pool_2 = nn.MaxPool1d(kernel_sizes[2], self.stride)
#         # fully connected layer.
#         self.fc = nn.Linear(self.in_features_fc(), output_embed_size)
#         # activation functions.
#         self.relu = nn.ReLU()
#         self.root_emb_size = self.embedding_size**0.5
#         self.sigmoid = nn.Sigmoid()
#         # self.softmax = nn.Softmax(dim=-1)
#     def out_pool_size(self, kernel_size, stride):
#         out_conv = ((self.embedding_size - 1 * (kernel_size - 1) - 1) / stride) + 1
#         out_conv = math.floor(out_conv)
#         out_pool = ((out_conv - 1 * (kernel_size - 1) - 1) / stride) + 1
#         out_pool = math.floor(out_pool)
        
#         return out_pool
        
#     def in_features_fc(self):
#         '''
#         Calculates the number of output features after Convolution + Max pooling
            
#         Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
#         Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
        
#         source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
#         '''
#         # Calcualte size of convolved/pooled features for convolution_{i}/max_pooling_{i} features
#         out_pool_sizes = []
#         for i in range(self.num_conv_layers):
#             out_pool_sizes.append(
#                 self.out_pool_size(
#                     self.kernel_sizes[i], 
#                     self.stride,
#                 )
#             )

#         return sum(out_pool_sizes) * self.out_size
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: text or code tokens.
#         # Sequence of tokes is filterd through an embedding layer
#         x = self.embedding(x)
#         # Convolution layer 1 is applied
#         x1 = self.conv_0(x)
#         x1 = self.relu(x1)
#         x1 = self.pool_0(x1)
#         # Convolution layer 2 is applied
#         x2 = self.conv_1(x)
#         x2 = self.relu(x2)
#         x2 = self.pool_1(x2)
#         # Convolution layer 3 is applied
#         x3 = self.conv_2(x)
#         x3 = self.relu(x3)
#         x3 = self.pool_2(x3)
#         # The output of each convolutional layer is concatenated into a unique vector
#         union = torch.cat((x1, x2, x3), 2)
#         # union = union.reshape(union.size(0), -1)
#         print("union.shape = ", union.shape)
#         # # The "flattened" vector is passed through a fully connected layer
#         out = self.fc(union)
#         # print("out:", out)
#         out = F.normalize(out, p=2, dim=-1)
#         # print("out:", out)
#         # out = self.tanh(out)
#         # # Dropout is applied		
#         # out = self.dropout(out)
#         # # Activation function is applied
#         # out = self.sigmoid(out)
#         return out