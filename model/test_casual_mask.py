import torch 

sequence_length= 4

mask = torch.tril(
    torch.ones((sequence_length, sequence_length))
)

print(mask)