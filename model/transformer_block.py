import torch
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .mlp import MLP


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1):
        super().__init__()

        # Layer normalization before attention
        self.ln1 = nn.LayerNorm(embed_dim)

        # Multi-head attention
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            block_size=block_size
        )

        # Layer normalization before MLP
        self.ln2 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.mlp = MLP(embed_dim)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):

        # Attention + residual connection
        x = x + self.dropout1(
            self.attention(self.ln1(x))
        )

        # MLP + residual connection
        x = x + self.dropout2(
            self.mlp(self.ln2(x))
        )

        return x
