import torch 
import torch.nn as nn

batch_size = 4
embed_dim = 256
sequence_length = 256

x = torch.randn(batch_size , sequence_length , embed_dim)

layer_norm = nn.LayerNorm(embed_dim)

normalized_x = layer_norm(x)

#simulating a transformation 
transformation = nn.Linear(embed_dim, embed_dim)
transformed_x = transformation(normalized_x)

output = transformed_x + x  # Residual connection

print("Input shape:", x.shape)
print("Normalized shape:", normalized_x.shape)
print("Transformed shape:", transformed_x.shape)
print("Output shape:", output.shape)

print("Normalized mean (approx):", normalized_x[0,0].mean().item())
print("Normalized std (approx):", normalized_x[0,0].std().item())

print("Output contains NaN:", torch.isnan(output).any().item())