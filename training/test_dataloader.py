
import torch
from torch.utils.data import DataLoader

from datasets import GPTDataset


BLOCK_SIZE = 256
BATCH_SIZE = 4


# Load training dataset
train_dataset = GPTDataset(
    "data/processed/train_tokens.npy",
    block_size=BLOCK_SIZE
)

# Load validation dataset
val_dataset = GPTDataset(
    "data/processed/validation_tokens.npy",
    block_size=BLOCK_SIZE
)


# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Get one training batch
x, y = next(iter(train_loader))


print("Training examples:", len(train_dataset))
print("Validation examples:", len(val_dataset))

print("\nInput shape:", x.shape)
print("Target shape:", y.shape)

print("\nFirst 10 input IDs:")
print(x[0][:10])

print("\nFirst 10 target IDs:")
print(y[0][:10])

print("\nChecking shifted relationship:")
print(torch.equal(x[0][1:], y[0][:-1]))
