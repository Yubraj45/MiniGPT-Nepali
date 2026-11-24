import torch
import torch.nn as nn

class GPTEmbeddings(nn.Module):
    def __init__(self , vocab_size , embed_dim , block_size):
        
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size , embed_dim)
        self.position_embeddings = nn.Embedding(block_size , embed_dim)
        
        nn.init.normal_(self.token_embeddings.weight , mean=0.0 , std= 0.02)  #-> fills a tensor with values drawn from a normal (Gaussian) distribution \(\mathcal{N}(\text{mean}, \text{std}^2)\)
        nn.init.normal_(self.position_embeddings.weight , mean=0.0 , std= 0.02)
        
    def forward(self , x):
        batch_size , seq_len = x.shape
        
        token_embeddings = self.token_embeddings(x)
        
        positions = torch.arange(
            seq_len , 
            device = x.device
        )
        
        position_embeddings = self.position_embeddings(positions)
        
        # combine
        
        embeddings = token_embeddings + position_embeddings
        
        return embeddings