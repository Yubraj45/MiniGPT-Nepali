import torch
from tokenizers import Tokenizer
from model.gpt import GPT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BLOCK_SIZE = 256
MAX_NEW_TOKENS = 100

TOKENIZER_PATH = "tokenizer/tokenizer_metaspace.json"

MODEL_PATHS = {
    "Original model": "checkpoints/best_model.pth",
    "Phase 1 expanded model": "checkpoints_expanded/best_model.pth",
    "Phase 2 expanded model": "checkpoints_expanded_phase2/best_model.pth"
}


def load_model(checkpoint_path):

    model = GPT().to(DEVICE)

    state_dict = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
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


def generate_text(
    model,
    tokenizer,
    prompt,
    strategy="top_p",
    top_k=50,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2,
    no_repeat_ngram_size=0
):

    input_ids = tokenizer.encode(prompt).ids

    idx = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=DEVICE
    )

    prompt_length = idx.size(1)

    with torch.no_grad():

        for _ in range(MAX_NEW_TOKENS):

            context = idx[:, -BLOCK_SIZE:]
            logits = model(context)
            logits = logits[:, -1, :]

            if no_repeat_ngram_size > 0:

                sequence = idx[0].tolist()

                banned_tokens = get_banned_tokens(
                    sequence,
                    no_repeat_ngram_size
                )

                if banned_tokens:
                    logits[0, banned_tokens] = float("-inf")

            if repetition_penalty != 1.0:

                generated_tokens = idx[0, prompt_length:].tolist()

                for token_id in set(generated_tokens):

                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # Apply temperature consistently before sampling
            if temperature <= 0:
                raise ValueError("Temperature must be greater than zero.")

            logits = logits / temperature

            if strategy == "greedy":

                next_token = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True
                )

            elif strategy == "top_p":

                sorted_logits, sorted_indices = torch.sort(
                    logits,
                    descending=True,
                    dim=-1
                )

                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p

                sorted_indices_to_remove[..., 1:] = (
                    sorted_indices_to_remove[..., :-1].clone()
                )

                sorted_indices_to_remove[..., 0] = False

                sorted_logits[sorted_indices_to_remove] = float("-inf")

                filtered_probs = torch.softmax(sorted_logits, dim=-1)

                sampled_sorted_index = torch.multinomial(
                    filtered_probs,
                    num_samples=1
                )

                next_token = torch.gather(
                    sorted_indices,
                    -1,
                    sampled_sorted_index
                )

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            idx = torch.cat((idx, next_token), dim=1)

            if next_token.item() == tokenizer.token_to_id("[EOS]"):
                break

    return tokenizer.decode(
        idx[0].tolist(),
        skip_special_tokens=True
    )


if __name__ == "__main__":

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    prompt = "नेपालको इतिहास"

    print("Prompt:", prompt)
    print("Decoding: Top-p | Temperature: 0.8 | Top-p: 0.9")
    print("Repetition penalty: 1.2")
    print("-" * 70)

    for model_name, checkpoint_path in MODEL_PATHS.items():

        model = load_model(checkpoint_path)

        # Reset sampling RNG for reproducibility across model runs
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        output = generate_text(
            model,
            tokenizer,
            prompt,
            strategy="top_p",
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2
        )

        print(f"\n{model_name}:")
        print(output)
        print("-" * 70)

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
