import torch
from .embeddings import GPTEmbeddings

vocab_size = 8000
embed_dim = 256
block_size = 256
batch_size = 4
sequence_length = 256

x = torch.randint(0 , 
                  vocab_size ,
                  (batch_size , sequence_length))

embedding_layer = GPTEmbeddings(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    block_size=block_size
)

output = embedding_layer(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Parameters :", sum(p.numel() for p in embedding_layer.parameters()))