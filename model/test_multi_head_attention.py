import torch 
from .multi_head_attention import MultiHeadAttention

batch_size = 4
sequence_length = 256
embed_dim = 256
num_heads = 4

x = torch.randn(batch_size , sequence_length , embed_dim)

attention = MultiHeadAttention(
    embed_dim=embed_dim,
    num_heads=num_heads,
    block_size=256
)

output = attention(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Output contains NaN:", torch.isnan(output).any().item())
print("Parameters in MultiHeadAttention:", sum(p.numel() for p in attention.parameters()))