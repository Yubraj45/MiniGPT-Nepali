import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.datasets import GPTDataset
from model.gpt import GPT


# CONFIGURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4
BLOCK_SIZE = 256
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
PATIENCE = 3

CHECKPOINT_DIR = "checkpoints_expanded"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# DATASETS
train_dataset = GPTDataset(
    "data/processed/train_expanded_tokens.npy",
    block_size=BLOCK_SIZE
)

val_dataset = GPTDataset(
    "data/processed/validation_tokens.npy",
    block_size=BLOCK_SIZE
)

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


# MODEL
model = GPT().to(DEVICE)

# Load epoch 9 best weights
state_dict = torch.load(
    "checkpoints_dropout/best_model.pth",
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(state_dict)

print("Loaded original best model successfully.")


# LOSS AND OPTIMIZER
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)


# TRAINING
def train_one_epoch(model, loader, optimizer, loss_fn):

    model.train()
    total_loss = 0.0

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)

        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item() * y.numel()

    return total_loss / (len(loader.dataset) * BLOCK_SIZE)


# VALIDATION
def validate(model, loader, loss_fn):

    model.eval()
    total_loss = 0.0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

            total_loss += loss.item() * y.numel()

    return total_loss / (len(loader.dataset) * BLOCK_SIZE)


# INITIAL VALIDATION
best_val_loss = validate(model, val_loader, loss_fn)

print(f"Starting validation loss: {best_val_loss:.4f}")

# Preserve initial model as the fallback best model
torch.save(
    model.state_dict(),
    f"{CHECKPOINT_DIR}/best_model.pth"
)

epochs_without_improvement = 0


# TRAINING LOOP
for epoch in range(NUM_EPOCHS):

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        loss_fn
    )

    val_loss = validate(
        model,
        val_loader,
        loss_fn
    )

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Validation Loss: {val_loss:.4f}"
    )

    improved = val_loss < best_val_loss

    if improved:

        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            f"{CHECKPOINT_DIR}/best_model.pth"
        )

        print("New best model saved!")

    else:
        epochs_without_improvement += 1

    # Save latest checkpoint
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss
    }

    torch.save(
        checkpoint,
        f"{CHECKPOINT_DIR}/latest_checkpoint.pth"
    )

    if epochs_without_improvement >= PATIENCE:

        print("Early stopping: validation loss did not improve.")
        break


print("Training experiment finished.")
print(f"Best validation loss: {best_val_loss:.4f}")
