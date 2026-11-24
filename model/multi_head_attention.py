import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.projection = nn.Linear(embed_dim, embed_dim)

        self.scale = math.sqrt(self.head_dim)

        self.attention_dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):

        batch_size, seq_len, embed_dim = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / self.scale

        # Apply causal mask
        scores = scores.masked_fill(
            self.mask[:seq_len, :seq_len] == 0,
            float("-inf")
        )

        attention_weights = torch.softmax(scores, dim=-1)

        # Attention dropout
        attention_weights = self.attention_dropout(attention_weights)

        output = attention_weights @ V

        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, embed_dim)
        )

        # Final projection
        output = self.projection(output)

        return output
