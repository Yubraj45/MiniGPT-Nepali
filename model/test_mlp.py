import torch 
from .mlp import MLP

batch_size = 4
embed_dim = 256
sequence_length = 256

x = torch.randn(batch_size , sequence_length , embed_dim)
mlp = MLP(embed_dim)
output = mlp(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Parameters :", sum(p.numel() for p in mlp.parameters()))
print("Output contains NaN:", torch.isnan(output).any().item())