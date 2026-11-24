import torch
from tokenizers import Tokenizer
from model.gpt import GPT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BLOCK_SIZE = 256
MAX_NEW_TOKENS = 100

MODEL_PATH = "checkpoints_expanded/best_model.pth"
TOKENIZER_PATH = "tokenizer/tokenizer_metaspace.json"

PROMPT = "नेपालको इतिहास"

CONFIGURATIONS = [
    {"name": "A", "temperature": 0.7, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0},
    {"name": "B", "temperature": 0.8, "repetition_penalty": 1.2, "no_repeat_ngram_size": 0},
    {"name": "C", "temperature": 0.8, "repetition_penalty": 1.5, "no_repeat_ngram_size": 0},
    {"name": "D", "temperature": 0.8, "repetition_penalty": 1.2, "no_repeat_ngram_size": 2},
    {"name": "E", "temperature": 0.8, "repetition_penalty": 1.2, "no_repeat_ngram_size": 3},
    {"name": "F", "temperature": 1.0, "repetition_penalty": 1.2, "no_repeat_ngram_size": 3},
]


def load_model():

    model = GPT().to(DEVICE)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def get_banned_tokens(sequence, ngram_size):

    if ngram_size <= 0:
        return []

    if len(sequence) < ngram_size - 1:
        return []

    current_prefix = tuple(sequence[-(ngram_size - 1):])
    banned_tokens = []

    for i in range(len(sequence) - ngram_size + 1):

        ngram = sequence[i:i + ngram_size]

        if tuple(ngram[:-1]) == current_prefix:
            banned_tokens.append(ngram[-1])

    return banned_tokens


def generate(model, tokenizer, prompt, config):

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
            logits = model(context)[:, -1, :]

            # Repetition penalty
            generated_tokens = idx[0, prompt_length:].tolist()

            for token_id in set(generated_tokens):

                if logits[0, token_id] > 0:
                    logits[0, token_id] /= config["repetition_penalty"]
                else:
                    logits[0, token_id] *= config["repetition_penalty"]

            # No-repeat n-gram
            ngram_size = config["no_repeat_ngram_size"]

            if ngram_size > 0:

                banned_tokens = get_banned_tokens(
                    idx[0].tolist(),
                    ngram_size
                )

                if banned_tokens:
                    logits[0, banned_tokens] = float("-inf")

            # Temperature
            logits = logits / config["temperature"]

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(
                logits,
                descending=True,
                dim=-1
            )

            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            remove_indices = cumulative_probs > 0.9

            remove_indices[..., 1:] = (
                remove_indices[..., :-1].clone()
            )

            remove_indices[..., 0] = False

            sorted_logits[remove_indices] = float("-inf")

            filtered_probs = torch.softmax(sorted_logits, dim=-1)

            sampled_index = torch.multinomial(
                filtered_probs,
                num_samples=1
            )

            next_token = torch.gather(
                sorted_indices,
                -1,
                sampled_index
            )

            idx = torch.cat((idx, next_token), dim=1)

            if next_token.item() == tokenizer.token_to_id("[EOS]"):
                break

    return tokenizer.decode(
        idx[0].tolist(),
        skip_special_tokens=True
    )


if __name__ == "__main__":

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    model = load_model()

    print("Model: Phase 1 expanded")
    print("Prompt:", PROMPT)

    for config in CONFIGURATIONS:

        # Reproducible sampling for each configuration
        torch.manual_seed(42)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        output = generate(
            model,
            tokenizer,
            PROMPT,
            config
        )

        print("\n" + "=" * 60)
        print("Experiment:", config["name"])
        print("Configuration:", config)
        print("Output:", output)

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
