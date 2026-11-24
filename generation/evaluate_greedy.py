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

PROMPTS = [
    "नेपालको इतिहास",
    "नेपालको शिक्षा प्रणाली",
    "कृत्रिम बुद्धिमत्ता",
    "नेपालको अर्थतन्त्र",
    "विज्ञान र प्रविधि"
]


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


def generate_greedy(model, tokenizer, prompt):

    input_ids = tokenizer.encode(prompt).ids

    idx = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():

        for _ in range(MAX_NEW_TOKENS):

            context = idx[:, -BLOCK_SIZE:]

            logits = model(context)
            logits = logits[:, -1, :]

            next_token = torch.argmax(
                logits,
                dim=-1,
                keepdim=True
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

    print("DEVICE:", DEVICE)
    print("DECODING: Greedy")
    print("MAX NEW TOKENS:", MAX_NEW_TOKENS)

    for model_name, checkpoint_path in MODEL_PATHS.items():

        print("\n" + "=" * 80)
        print(model_name)
        print("=" * 80)

        model = load_model(checkpoint_path)

        for prompt in PROMPTS:

            output = generate_greedy(
                model,
                tokenizer,
                prompt
            )

            print(f"\nPROMPT: {prompt}")
            print(f"OUTPUT: {output}")

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
