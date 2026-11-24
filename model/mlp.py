import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.network(x)
