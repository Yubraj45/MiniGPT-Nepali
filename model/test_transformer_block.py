import torch 
from .transformer_block import TransformerBlock

batch_size = 4
sequence_length = 256
embed_dim = 256
num_heads =4
block_size = 256

x = torch.rand(batch_size, sequence_length, embed_dim)

block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, block_size=block_size)

output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Parameters:" , sum(p.numel() for p in block.parameters() if p.requires_grad))
print("Output contains NaN values:", torch.isnan(output).any().item())                                                              