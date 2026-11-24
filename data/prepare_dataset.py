
from datasets import load_dataset
from pathlib import Path

DATASET_NAME = "Titung/cc100-nepali-cleaned"

TRAIN_LIMIT = 20_000
VALIDATION_LIMIT = 1_000

MIN_WORDS = 5
MIN_DEVANAGARI_RATIO = 0.80

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)


def prepare_split(split, limit, output_file):

    dataset = load_dataset(
        DATASET_NAME,
        split=split,
        streaming=True
    )

    collected = 0
    rejected = 0

    with open(output_file, "w", encoding="utf-8") as f:

        for sample in dataset:

            text = sample["text"].strip()

            # Basic quality checks
            if not text:
                rejected += 1
                continue

            if sample["word_count"] < MIN_WORDS:
                rejected += 1
                continue

            if sample["devanagari_ratio"] < MIN_DEVANAGARI_RATIO:
                rejected += 1
                continue

            # Keeping each document on one   line
            text = " ".join(text.split())

            f.write(text + "\n")

            collected += 1

            if collected % 1000 == 0:
                print(f"{split}: collected {collected}")

            if collected >= limit:
                break

    print(f"\nFinished {split}")
    print("Collected:", collected)
    print("Rejected:", rejected)


prepare_split(
    "train",
    TRAIN_LIMIT,
    output_dir / "train.txt"
)

prepare_split(
    "validation",
    VALIDATION_LIMIT,
    output_dir / "validation.txt"
)

print("\nDataset preparation completed!")




from datasets import load_dataset
from pathlib import Path
import re
import html

DATASET_NAME = "Titung/cc100-nepali-cleaned"

TRAIN_LIMIT = 20_000
VALIDATION_LIMIT = 1_000

MIN_WORDS = 5
MIN_DEVANAGARI_RATIO = 0.80
MAX_TEXT_LENGTH = 10_000

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text):

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace
    text = " ".join(text.split())

    return text


def prepare_split(split, limit, output_file):

    dataset = load_dataset(
        DATASET_NAME,
        split=split,
        streaming=True
    )

    collected = 0
    rejected = 0
    seen = set()

    with open(output_file, "w", encoding="utf-8") as f:

        for sample in dataset:

            text = clean_text(sample["text"])

            if not text:
                rejected += 1
                continue

            if "\ufffd" in text:
                rejected += 1
                continue

            if len(text) > MAX_TEXT_LENGTH:
                rejected += 1
                continue

            if len(text.split()) < MIN_WORDS:
                rejected += 1
                continue

            if sample["devanagari_ratio"] < MIN_DEVANAGARI_RATIO:
                rejected += 1
                continue

            # Remove exact duplicates
            if text in seen:
                rejected += 1
                continue

            seen.add(text)

            f.write(text + "\n")
            collected += 1

            if collected % 1000 == 0:
                print(f"{split}: {collected} documents")

            if collected >= limit:
                break

    print(f"\n{split} completed")
    print("Collected:", collected)
    print("Rejected:", rejected)


prepare_split(
    "train",
    TRAIN_LIMIT,
    OUTPUT_DIR / "train.txt"
)

prepare_split(
    "validation",
    VALIDATION_LIMIT,
    OUTPUT_DIR / "validation.txt"
)

print("\nDataset preparation completed!")
