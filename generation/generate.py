
import argparse
import torch
from tokenizers import Tokenizer
from model.gpt import GPT



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BLOCK_SIZE = 256
TOKENIZER_PATH = "tokenizer/tokenizer_metaspace.json"

MODEL_PATHS = {
    "original": "checkpoints/best_model.pth",
    "dropout": "checkpoints_dropout/best_model.pth",
    "phase1": "checkpoints_expanded/best_model.pth",
    "phase2": "checkpoints_expanded_phase2/best_model.pth",
}



def load_tokenizer():
    return Tokenizer.from_file(TOKENIZER_PATH)


def load_model(model_name):
    if model_name not in MODEL_PATHS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_PATHS.keys())}"
        )

    model = GPT().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATHS[model_name],
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model



def get_banned_tokens(sequence, ngram_size):

    if ngram_size <= 0:
        return []

    if ngram_size == 1:
        return list(set(sequence))

    if len(sequence) < ngram_size - 1:
        return []

    current_prefix = tuple(sequence[-(ngram_size - 1):])

    banned_tokens = []

    for i in range(len(sequence) - ngram_size + 1):

        ngram = sequence[i:i + ngram_size]

        if tuple(ngram[:-1]) == current_prefix:
            banned_tokens.append(ngram[-1])

    return banned_tokens


def apply_repetition_penalty(logits, sequence, prompt_length, penalty):

    if penalty == 1.0:
        return logits

    generated_tokens = sequence[prompt_length:]

    for token_id in set(generated_tokens):

        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty

    return logits



def sample_next_token(
    logits,
    strategy,
    temperature,
    top_k,
    top_p
):

    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    # Apply temperature consistently before sampling
    logits = logits / temperature

    if strategy == "greedy":

        return torch.argmax(
            logits,
            dim=-1,
            keepdim=True
        )

    elif strategy == "temperature":

        probabilities = torch.softmax(logits, dim=-1)

        return torch.multinomial(
            probabilities,
            num_samples=1
        )

    elif strategy == "top_k":

        if top_k < 1 or top_k > logits.size(-1):
            raise ValueError("top_k must be between 1 and vocab size.")

        top_k_logits, top_k_indices = torch.topk(
            logits,
            k=top_k,
            dim=-1
        )

        probabilities = torch.softmax(
            top_k_logits,
            dim=-1
        )

        sampled_index = torch.multinomial(
            probabilities,
            num_samples=1
        )

        return torch.gather(
            top_k_indices,
            -1,
            sampled_index
        )

    elif strategy == "top_p":

        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1].")

        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
            dim=-1
        )

        sorted_probs = torch.softmax(
            sorted_logits,
            dim=-1
        )

        cumulative_probs = torch.cumsum(
            sorted_probs,
            dim=-1
        )

        remove_indices = cumulative_probs > top_p

        remove_indices[..., 1:] = (
            remove_indices[..., :-1].clone()
        )

        remove_indices[..., 0] = False

        sorted_logits[remove_indices] = float("-inf")

        filtered_probs = torch.softmax(
            sorted_logits,
            dim=-1
        )

        sampled_index = torch.multinomial(
            filtered_probs,
            num_samples=1
        )

        return torch.gather(
            sorted_indices,
            -1,
            sampled_index
        )

    else:

        raise ValueError(
            f"Unknown generation strategy: {strategy}"
        )



def generate_text(
    model,
    tokenizer,
    prompt,
    strategy="top_p",
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,
    seed=None
):

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1.")

    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive.")

    if no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size cannot be negative.")

    if seed is not None:
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    input_ids = tokenizer.encode(prompt).ids

    if not input_ids:
        raise ValueError("Prompt must contain at least one token.")

    idx = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=DEVICE
    )

    prompt_length = idx.size(1)

    eos_id = tokenizer.token_to_id("[EOS]")

    model.eval()

    with torch.no_grad():

        for _ in range(max_new_tokens):

            context = idx[:, -BLOCK_SIZE:]

            logits = model(context)[:, -1, :]

            sequence = idx[0].tolist()

            # No-repeat n-gram blocking
            if no_repeat_ngram_size > 0:

                banned_tokens = get_banned_tokens(
                    sequence,
                    no_repeat_ngram_size
                )

                if banned_tokens:
                    logits[0, banned_tokens] = float("-inf")

            # Repetition penalty
            logits = apply_repetition_penalty(
                logits,
                sequence,
                prompt_length,
                repetition_penalty
            )

            next_token = sample_next_token(
                logits,
                strategy,
                temperature,
                top_k,
                top_p
            )

            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

            if next_token.item() == eos_id:
                break

    return tokenizer.decode(
        idx[0].tolist(),
        skip_special_tokens=True
    )



def main():

    parser = argparse.ArgumentParser(
        description="Generate Nepali text using MiniGPT-Nepali."
    )

    parser.add_argument(
        "--model",
        choices=MODEL_PATHS.keys(),
        default="phase1"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="नेपालको इतिहास"
    )

    parser.add_argument(
        "--strategy",
        choices=["greedy", "temperature", "top_k", "top_p"],
        default="top_p"
    )

    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    tokenizer = load_tokenizer()
    model = load_model(args.model)

    print("=" * 60)
    print("MiniGPT-Nepali")
    print("=" * 60)

    print("Model:", args.model)
    print("Device:", DEVICE)
    print("Strategy:", args.strategy)
    print("Prompt:", args.prompt)

    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        strategy=args.strategy,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        seed=args.seed
    )

    print("\nGenerated text:")
    print("-" * 60)
    print(output)
    print("-" * 60)


if __name__ == "__main__":
    main()

