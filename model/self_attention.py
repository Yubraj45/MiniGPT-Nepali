
import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, block_size):
        super().__init__()

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.scale = math.sqrt(embed_dim)

        # Causal masking so  that each token can only see itself and previous tokens
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, embed_dim)

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Calculating attention scores
        scores = Q @ K.transpose(-2, -1)
        scores = scores / self.scale

        # Applying  causal mask
        seq_len = x.size(1)
        scores = scores.masked_fill(
            self.mask[:seq_len, :seq_len] == 0,
            float("-inf")
        )

        # Convert scores into probabilities
        attention_weights = torch.softmax(scores, dim=-1)

        # Weighted combination of values
        output = attention_weights @ V

        return output
