from datasets import load_dataset
import random

dataset = load_dataset(
    "Titung/cc100-nepali-cleaned",
    split="train",
    streaming=True
)

# Shuffle the stream with a fixed seed
dataset = dataset.shuffle(
    seed=42,
    buffer_size=1000
)

for i, sample in enumerate(dataset):

    print(f"\nExample {i + 1}")
    print("Text:", sample["text"])
    print("Words:", sample["word_count"])
    print("Devanagari ratio:", sample["devanagari_ratio"])

    if i == 49:
        break
    
    
from pathlib import Path

files = [
    Path("data/processed/train.txt"),
    Path("data/processed/validation.txt")
]

for file in files:
    print(f"\n{'=' * 50}")
    print("File:", file)

    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print("Number of documents:", len(lines))

    for i, line in enumerate(lines[:5]):
        print(f"\nDocument {i + 1}:")
        print(line.strip())