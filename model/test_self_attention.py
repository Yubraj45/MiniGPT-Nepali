
import torch
from .self_attention import SelfAttention

batch_size = 4
sequence_length = 256
embed_dim = 256

x = torch.randn(batch_size, sequence_length, embed_dim)

attention = SelfAttention(
    embed_dim=embed_dim,
    block_size=256
)

output = attention(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Output contains NaN:", torch.isnan(output).any().item())
