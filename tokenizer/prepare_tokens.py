
from tokenizers import Tokenizer
from pathlib import Path
import numpy as np

#loading the tokenizer meta_space
tokenizer = Tokenizer.from_file(
    "tokenizer/tokenizer_metaspace.json"
)

eos_id = tokenizer.token_to_id("[EOS]")

if eos_id is None:
    raise ValueError("[EOS] token not found!")


def tokenize_file(input_file, output_file):

    all_tokens = []
    document_count = 0

    with open(input_file, "r", encoding="utf-8") as f:

        for line in f:

            text = line.strip()

            if not text:
                continue

            # Converting text into token IDs
            token_ids = tokenizer.encode(text).ids

            # Appending document tokens
            all_tokens.extend(token_ids)

            # Marking the end of this document
            all_tokens.append(eos_id)

            document_count += 1

            if document_count % 5000 == 0:
                print(
                    f"Processed {document_count} documents"
                )

    # Store IDs using unsigned 16-bit integers
    token_array = np.array(
        all_tokens,
        dtype=np.uint16
    )

    np.save(output_file, token_array)

    print("\nCompleted:", input_file)
    print("Documents:", document_count)
    print("Total tokens:", len(token_array))
    print("Saved:", output_file)


# Training dataset
tokenize_file(
    "data/processed/train.txt",
    "data/processed/train_tokens.npy"
)

# Validation dataset
tokenize_file(
    "data/processed/validation.txt",
    "data/processed/validation_tokens.npy"
)

print("\nTokenization completed!")
