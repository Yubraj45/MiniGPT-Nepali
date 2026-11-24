from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.normalizers import NFC
from tokenizers.decoders import Metaspace as MetaspaceDecoder

from pathlib import Path

TRAIN_FILE = Path("data/processed/train.txt")
OUTPUT_DIR = Path("tokenizer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_SIZE = 8000

SPECIAL_TOKENS = [
    "[UNK]",
    "[PAD]",
    "[BOS]",
    "[EOS]"
]


tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.normalizer = NFC()

tokenizer.pre_tokenizer = Metaspace()


trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
    min_frequency=2,
    show_progress=True
)

tokenizer.train(
    files=[str(TRAIN_FILE)],
    trainer=trainer
)

tokenizer.decoder = MetaspaceDecoder()
tokenizer.save(str(OUTPUT_DIR / "tokenizer_metaspace.json"))

test_text = "म नेपाली भाषा प्रयोग गरेर आफ्नै GPT मोडेल बनाउँदैछु।"

encoded = tokenizer.encode(test_text)

print("\nOriginal text:")
print(test_text)

print("\nTokens:")
print(encoded.tokens)

print("\nToken IDs:")
print(encoded.ids)

print("\nDecoded text:")
print(tokenizer.decode(encoded.ids))