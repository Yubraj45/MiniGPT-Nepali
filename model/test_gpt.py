import torch 
from .gpt import GPT

batch_size = 4
sequence_length = 256

x = torch.randint(0, 8000, (batch_size, sequence_length))

model = GPT()
logits = model(x)

total_params = sum(
    p.numel() for p in model.parameters()
)

print("Input shape:", x.shape)
print("Logits shape:" , logits.shape)
print("Total parameters:" , total_params)
print("Output contains NaN:",torch.isnan(logits).any().item())