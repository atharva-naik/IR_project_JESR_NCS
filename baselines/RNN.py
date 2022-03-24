import torch
import torch.nn as nn


class RNNEncoder(nn.Module):
    def __init__(self):
        super(RNNEncoder, self).__init__()
        dropout = args.get("dropout", 0.2)
        seq_len = args.get("seq_len", 100)
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