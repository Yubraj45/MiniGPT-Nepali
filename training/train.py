import os 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.datasets import GPTDataset
from model.gpt import GPT

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# configuraiton
BATCH_SIZE = 4
BLOCK_SIZE = 256
LEARNING_RATE = 1e-4


# LOADING TRAINING DATASET
train_dataset = GPTDataset(
    "data/processed/train_tokens.npy",
    block_size=BLOCK_SIZE
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# LOADING VALIDATION DATASET
val_dataset = GPTDataset(
    "data/processed/validation_tokens.npy",
    block_size=BLOCK_SIZE
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Creating model
model = GPT().to(DEVICE)

# loss fucntion
loss_fn = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)


def train_one_epoch(model, train_loader, optimizer, loss_fn):

    model.train()

    total_loss = 0.0

    for x, y in train_loader:
        x , y = x.to(DEVICE) , y.to(DEVICE)
        # forward pass
        logits = model(x)

        # calculate the loss
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1)
        )

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        #gradient clipping --> Limits the overall gradient morm before the optimizer updates the weights 
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )
        

        optimizer.step()

        total_loss += loss.item() * y.numel()

    average_loss = total_loss/len(train_loader.dataset) /BLOCK_SIZE
    return average_loss


def validate(model, val_loader, loss_fn):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x , y = x.to(DEVICE) , y.to(DEVICE)
            logits = model(x)

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

            total_loss += loss.item() * y.numel()

    average_loss = total_loss/len(val_loader.dataset) / BLOCK_SIZE
    return average_loss



#saving the model checkpoint 
def save_checkpoint(model , optimizer , epoch , train_loss , val_loss , best_val_loss):
    checkpoint= {
        "epoch":epoch,
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss
    }
    
    os.makedirs("checkpoints",exist_ok=True )
    
    torch.save(
        checkpoint,
        "checkpoints/latest_checkpoint.pth"
    )
    print("Checkpoint saved.")


start_epoch=0
best_val_loss = float("inf")

def load_checkpoint(model, optimizer, checkpoint_path):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"]

    best_val_loss = checkpoint.get(
        "best_val_loss",
        checkpoint["val_loss"]
    )

    print(f"Resuming from epoch {start_epoch}")
    print(f"Previous validation loss: {best_val_loss:.4f}")

    return start_epoch, best_val_loss



NUM_EPOCHS = 20

#set true only when you want to resume an exextiung run 
RESUME = True

if RESUME:
    start_epoch , best_val_loss = load_checkpoint(
    model=model ,
    optimizer=optimizer,
    checkpoint_path="checkpoints/latest_checkpoint.pth"    
    )

    for param_group in optimizer.param_groups:
        param_group["lr"] = LEARNING_RATE

    print("Updated learning rate:", optimizer.param_groups[0]["lr"])

os.makedirs("checkpoints" , exist_ok=True)
 
for epoch in range(start_epoch, NUM_EPOCHS):
    train_loss = train_one_epoch(
        model, train_loader, optimizer=optimizer, loss_fn=loss_fn
    )

    val_loss = validate(
        val_loader=val_loader,
        model=model,
        loss_fn=loss_fn
    )
    
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Validation Loss: {val_loss:.4f}"
    )
    
    
    if val_loss <best_val_loss:
        best_val_loss = val_loss
        
        torch.save(
            model.state_dict(),
            "checkpoints/best_model.pth"
        )
        
        print("Best model saved!")
        
    save_checkpoint(model , optimizer=optimizer , epoch=epoch+1 , train_loss= train_loss , val_loss=val_loss , best_val_loss=best_val_loss)
        

