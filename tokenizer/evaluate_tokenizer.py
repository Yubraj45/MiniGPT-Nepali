
from tokenizers import Tokenizer
from pathlib import Path

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

train_file = Path("data/processed/train.txt")

total_documents = 0
total_words = 0
total_tokens = 0

with open(train_file, "r", encoding="utf-8") as f:

    for line in f:
        text = line.strip()

        if not text:
            continue

        encoding = tokenizer.encode(text)

        total_documents += 1
        total_words += len(text.split())
        total_tokens += len(encoding.ids)


print("Documents:", total_documents)
print("Words:", total_words)
print("Tokens:", total_tokens)

print("\nAverage words per document:", total_words / total_documents)
print("Average tokens per document:", total_tokens / total_documents)
print("Average tokens per word:", total_tokens / total_words)





from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

samples = [
    "नेपाल एउटा सुन्दर देश हो।",
    "म नेपाली भाषा प्रयोग गरेर आफ्नै GPT मोडेल बनाउँदैछु।",
    "कृत्रिम बुद्धिमत्ता र मेसिन लर्निङ कम्प्युटर विज्ञानका महत्वपूर्ण क्षेत्र हुन्।",
    "आज नेपालको अर्थतन्त्र र प्रविधिको विकासबारे छलफल भइरहेको छ।"
]

for text in samples:
    encoding = tokenizer.encode(text)

    words = len(text.split())
    tokens = len(encoding.ids)

    print("\nText:", text)
    print("Tokens:", encoding.tokens)
    print("Token count:", tokens)
    print("Word count:", words)
    print("Tokens per word:", round(tokens / words, 2))





from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

total_tokens = 0
total_documents = 0

with open("data/processed/validation.txt", "r", encoding="utf-8") as f:
    for line in f:
        text = line.strip()

        if text:
            total_tokens += len(tokenizer.encode(text).ids)
            total_documents += 1

print("Validation documents:", total_documents)
print("Validation tokens:", total_tokens)
