
import numpy as np

train = np.load(
    "data/processed/train_tokens.npy"
)

validation = np.load(
    "data/processed/validation_tokens.npy"
)

print("Training tokens:", len(train))
print("Validation tokens:", len(validation))

print("\nTraining dtype:", train.dtype)
print("Validation dtype:", validation.dtype)

print("\nFirst 30 training token IDs:")
print(train[:30])

print("\nMinimum token ID:", train.min())
print("Maximum token ID:", train.max())
