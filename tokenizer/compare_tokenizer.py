
from tokenizers import Tokenizer

byte_tokenizer = Tokenizer.from_file(
    "tokenizer/tokenizer.json"
)

metaspace_tokenizer = Tokenizer.from_file(
    "tokenizer/tokenizer_metaspace.json"
)

files = {
    "train": "data/processed/train.txt",
    "validation": "data/processed/validation.txt"
}

for split, file_path in files.items():

    byte_tokens = 0
    metaspace_tokens = 0
    words = 0
    documents = 0

    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:

            text = line.strip()

            if not text:
                continue

            byte_tokens += len(
                byte_tokenizer.encode(text).ids
            )

            metaspace_tokens += len(
                metaspace_tokenizer.encode(text).ids
            )

            words += len(text.split())
            documents += 1

    print(f"\n{split.upper()}")
    print("Documents:", documents)
    print("Words:", words)

    print("ByteLevel tokens:", byte_tokens)
    print("Metaspace tokens:", metaspace_tokens)

    print(
        "ByteLevel tokens/word:",
        round(byte_tokens / words, 3)
    )

    print(
        "Metaspace tokens/word:",
        round(metaspace_tokens / words, 3)
    )

    reduction = (
        1 - metaspace_tokens / byte_tokens
    ) * 100

    print(
        "Token reduction:",
        round(reduction, 2),
        "%"
    )
