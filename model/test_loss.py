import torch
import torch.nn as nn
import numpy as np

from .gpt import GPT

data = np.load("data/processed/train_tokens.npy")

# creating one batch of 4 sequence
batch_size = 4
block_size = 256

start = 0
chunk = data[start:start+batch_size*block_size+1]

chunk = torch.tensor(chunk.astype(np.int64))


##view reshape the tensor intpo a multi-dimentional grid without changung the underlying data 
x = chunk[:-1].view(batch_size , block_size)
y = chunk[1:].view(batch_size , block_size)
##--> chunk[1:] because y is the target and we want to shift the sequence by 1 


#creating model
model = GPT()


#forward pass
logits = model(x)


print("Logits mean:", logits.mean().item())
print("Logits std:", logits.std().item())
print("Logits max:", logits.max().item())
print("Logits min:", logits.min().item())

#calculate loss 
loss_fn = nn.CrossEntropyLoss()

loss = loss_fn(
    logits.reshape(-1 , logits.size(-1)),
    y.reshape(-1)
)

print("Input shape:" ,x.shape)
print("Target shape:", y.shape)
print("Logits shape:", logits.shape)
print("Loss:", loss.item())
print("Loss contains NaN:", torch.isnan(loss).item())