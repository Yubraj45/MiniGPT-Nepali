import torch
from tokenizers import Tokenizer
from model.gpt import GPT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BLOCK_SIZE = 256
MAX_NEW_TOKENS = 80

TOKENIZER_PATH = "tokenizer/tokenizer_metaspace.json"

MODEL_PATHS = {
    "Original model": "checkpoints/best_model.pth",
    "Phase 1 expanded": "checkpoints_expanded/best_model.pth",
    "Phase 2 expanded": "checkpoints_expanded_phase2/best_model.pth"
}

PROMPTS = [
    "नेपालको इतिहास",
    "नेपालको शिक्षा प्रणाली",
    "कृत्रिम बुद्धिमत्ता",
    "नेपालको अर्थतन्त्र",
    "विज्ञान र प्रविधि"
]


def load_model(path):
    model = GPT().to(DEVICE)

    state_dict = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def generate(model, tokenizer, prompt):

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
                    logits[0, token_id] /= 1.2
                else:
                    logits[0, token_id] *= 1.2

            # Temperature
            logits = logits / 0.8

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

    print("Decoding: Top-p")
    print("Temperature: 0.8 | Top-p: 0.9 | Repetition penalty: 1.2")

    for name, path in MODEL_PATHS.items():

        model = load_model(path)

        print("\n" + "=" * 75)
        print(name)
        print("=" * 75)

        for i, prompt in enumerate(PROMPTS):

            # Same random seed for each model and prompt
            torch.manual_seed(42 + i)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42 + i)

            output = generate(model, tokenizer, prompt)

            print(f"\nPrompt: {prompt}")
            print(f"Output: {output}")

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
