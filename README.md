
# MiniGPT-Nepali 🇳🇵

A small GPT-style language model trained from scratch to generate Nepali text using PyTorch.

MiniGPT-Nepali is an educational project exploring the fundamentals of Large Language Model (LLM) engineering. The project implements a decoder-only Transformer, trains a tokenizer, prepares a Nepali text dataset, and trains a language model using causal next-token prediction.

The model was built from scratch without pretrained weights.

## Project Overview

- **Architecture:** Decoder-only Transformer (GPT-style)
- **Framework:** PyTorch
- **Language:** Nepali
- **Tokenizer:** BPE with Metaspace pre-tokenization
- **Vocabulary size:** 8,000
- **Parameters:** Approximately 5.27 million
- **Training:** From scratch
- **Hardware used:** Kaggle GPU (Tesla T4)

## Model Architecture

The model follows a GPT-style decoder-only Transformer architecture.

| Component | Configuration |
|---|---|
| Vocabulary size | 8,000 |
| Embedding dimension | 256 |
| Transformer layers | 4 |
| Attention heads | 4 |
| Head dimension | 64 |
| Context length | 256 tokens |
| MLP hidden dimension | 1,024 |
| Dropout | 0.1 |
| Positional embeddings | Learned |
| Normalization | LayerNorm |
| Output head | Weight-tied language modeling head |

### Architecture details

- Token and positional embeddings
- Multi-head causal self-attention
- Pre-LayerNorm Transformer blocks
- Feed-forward MLP layers
- Residual connections
- Dropout regularization
- Final LayerNorm
- Weight tying between token embeddings and the language modeling head

The model predicts the next token based on previously observed tokens.

## Dataset

The Nepali text corpus was obtained from Hugging Face:

**Source:** [Titung/cc100-nepali-cleaned](https://huggingface.co/datasets/Titung/cc100-nepali-cleaned)

The dataset was streamed and processed before training.

### Dataset preparation

The preprocessing pipeline included:

- HTML entity decoding and HTML tag removal
- Whitespace normalization
- Filtering documents with fewer than 5 words
- Filtering documents with less than 80% Devanagari characters
- Removing documents containing replacement characters
- Limiting document length to 10,000 characters
- Removing exact duplicates
- Appending EOS tokens to document boundaries

### Dataset versions

| Dataset | Training documents |
|---|---:|
| Original | 20,000 |
| Expanded | 100,000 |
| Validation | 1,000 |

The expanded dataset contains the original 20,000 documents and 80,000 additional unique documents.

The validation dataset remained unchanged across experiments to make comparisons more consistent.

## Tokenizer

A custom BPE tokenizer was trained using the Hugging Face `tokenizers` library.

Configuration:

- Vocabulary size: 8,000
- Minimum frequency: 2
- Normalization: NFC
- Pre-tokenization: Metaspace
- Special tokens: `[UNK]`, `[PAD]`, `[BOS]`, `[EOS]`

The final model uses:

`tokenizer/tokenizer_metaspace.json`

## Training

The model was trained using causal next-token prediction.

### Training configuration

- Optimizer: AdamW
- Initial learning rate: 3e-4
- Batch size: 4
- Sequence length: 256
- Gradient clipping: 1.0
- Loss function: Cross-entropy
- Regularization: Dropout and weight decay

### Training experiments

| Experiment | Description | Best validation loss |
|---|---|---:|
| Original | Initial training on 20K documents | 5.8170 |
| Dropout | Continued training with dropout | 5.7874 |
| Expanded Phase 1 | Fine-tuning on 100K documents | 4.8147 |
| Expanded Phase 2 | Continued training with a lower learning rate | 4.7905 |

The expanded training stages initialized from previous best checkpoints rather than restarting from random weights.

Phase 1 was selected as the default generation checkpoint based on qualitative comparisons. Phase 2 achieved a slightly lower validation loss but did not consistently produce better generated text.

## Text Generation

The inference pipeline supports four decoding strategies:

1. **Greedy decoding** — selects the highest-probability next token.
2. **Temperature sampling** — adjusts the probability distribution before sampling.
3. **Top-k sampling** — samples from the k most likely tokens.
4. **Top-p sampling** — samples from a dynamically selected probability distribution.

Additional generation controls:

- Temperature
- Top-k
- Top-p
- Repetition penalty
- No-repeat n-gram blocking
- Maximum generation length
- EOS-based stopping

### Example

Prompt:

`नेपालको आर्थिक विकासका लागि`

Example generated output:

> नेपालको आर्थिक विकासका लागि नै महत्वपूर्ण र समृद्ध नेपाल सुखी नेपालीको अवधारणा आएको छ ।

This is an example of model output, not a manually written completion.

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/yubraj45/MiniGPT-Nepali.git
cd MiniGPT-Nepali
```

### 2. Install dependencies

```bash
pip install torch tokenizers numpy
```

Additional dependencies may be required for dataset preparation and training.

### 3. Generate Nepali text

The default model is Phase 1.

```bash
python -m generation.generate
```

You can customize generation:

```bash
python -m generation.generate \
    --model phase1 \
    --prompt "नेपालको इतिहास" \
    --strategy top_p \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_new_tokens 100
```

Available model choices:

- `original`
- `dropout`
- `phase1`
- `phase2`

Available generation strategies:

- `greedy`
- `temperature`
- `top_k`
- `top_p`

## Project Structure

```text
MiniGPT-Nepali/
│
├── checkpoints_expanded/
│   └── best_model.pth
│
├── data/
│   ├── inspect_data.py
│   └── prepare_dataset.py
│
├── generation/
│   ├── generate.py
│   ├── compare_models.py
│   ├── compare_three_models.py
│   ├── decoding_experiments.py
│   ├── evaluate_greedy.py
│   └── evaluate_top_p.py
│
├── model/
│   ├── embeddings.py
│   ├── gpt.py
│   ├── mlp.py
│   ├── multi_head_attention.py
│   ├── self_attention.py
│   └── transformer_block.py
│
├── tokenizer/
│   ├── tokenizer_metaspace.json
│   ├── train_metaspace_tokenizer.py
│   └── evaluate_tokenizer.py
│
├── training/
│   ├── datasets.py
│   ├── train.py
│   ├── train_dropout.py
│   ├── train_expanded.py
│   └── train_expanded_phase2.py
│
├── .gitignore
└── README.md
```

## Limitations

MiniGPT-Nepali is an educational language model, not a production-ready Nepali language model.

Some limitations include:

- Relatively small model size (approximately 5.27M parameters).
- Limited training corpus compared with modern LLMs.
- No pretrained language representations.
- Short context window of 256 tokens.
- Occasional repetition, topic drift, and grammatical inconsistencies.
- Dataset noise inherited from the source corpus.

The purpose of this project was to understand and implement the core components of GPT-style language models rather than achieve state-of-the-art Nepali text generation.

## What I Learned

Through this project, I explored:

- Building a tokenizer from scratch using BPE.
- Preparing and tokenizing a text corpus.
- Implementing causal self-attention and multi-head attention.
- Building Transformer decoder blocks.
- Implementing autoregressive next-token prediction.
- Training language models using PyTorch.
- Working with checkpoints and validation loss.
- Experimenting with sampling and decoding strategies.
- Comparing training runs and analyzing generated text.

## Acknowledgements

- [PyTorch](https://pytorch.org/)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [CC-100 Nepali Cleaned Dataset](https://huggingface.co/datasets/Titung/cc100-nepali-cleaned)

---

**Built as a hands-on project to learn the fundamentals of LLM engineering.**
