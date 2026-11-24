import torch
import torch.nn as nn

from .embeddings import GPTEmbeddings
from .transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size=8000,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        block_size=256,
        ):
        super().__init__()

        self.block_size = block_size

        # token + position embedding
        self.embeddings =GPTEmbeddings(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            block_size=block_size
        )
        
        
        ##stecking transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                block_size=block_size
            )
            for _ in range(num_layers)
        ])
        
        #Final LayerNorm
        self.ln_f = nn.LayerNorm(embed_dim)
        
        #vocabulary prediction head
        self.lm_head = nn.Linear(
            embed_dim, vocab_size, bias=False
        )
        
        #weight tying 
        self.lm_head.weight = self.embeddings.token_embeddings.weight
    
    def forward(self, x):
        x = self.embeddings(x)
        
        #pass through transformer block
        for blocks in self.blocks:
            x = blocks(x)
        
        
        #final normalization
        x = self.ln_f(x)
        
        #PREDICT  vocabulary logits 
        logits = self.lm_head(x)
        return logits