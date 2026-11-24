
import torch
from tokenizers import Tokenizer
from model.gpt import GPT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BLOCK_SIZE = 256
MAX_NEW_TOKENS = 100

MODEL_PATH = "checkpoints/best_model.pth"
TOKENIZER_PATH = "tokenizer/tokenizer_metaspace.json"


def load_model():
    model = GPT().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


def get_banned_tokens(sequence , ngram_size):
  if ngram_size<=0:
    return []

  if ngram_size == 1:
    return list(set(sequence))
  
  if len(sequence) < ngram_size - 1 :
    return []

  #last n-1 tokens form the current prefix 
  current_prefix = tuple(sequence[-(ngram_size -1 ):])
  banned_tokens = []

  #searching for previously occuring n-grams 
  for i in range(len(sequence) - ngram_size +1):
    ngram = sequence[i:i+ ngram_size]

    #checking whether its prefix matches the currrent prefix 
    if tuple(ngram[:-1]) == current_prefix:

      # the final token would repeat an existing n-gram
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
    repetition_penalty=1.0,
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

            #no-repeat n-gram blocking 
            if no_repeat_ngram_size >0 :
              sequence = idx[0].tolist()

              banned_tokens = get_banned_tokens(
                sequence, 
                no_repeat_ngram_size
              )

              if banned_tokens:
                logits[0, banned_tokens] = float("-inf")

            # Repetition penalty
            if repetition_penalty != 1.0:

                generated_tokens = idx[0, prompt_length:].tolist()

                for token_id in set(generated_tokens):

                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # Greedy decoding
            if strategy == "greedy":

                next_token = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True
                )

            # Temperature sampling
            elif strategy == "temperature":

                scaled_logits = logits / temperature

                probs = torch.softmax(
                    scaled_logits,
                    dim=-1
                )

                next_token = torch.multinomial(
                    probs,
                    num_samples=1
                )

            # Top-k sampling
            elif strategy == "top_k":

                top_k_logits, top_k_indices = torch.topk(
                    logits,
                    k=top_k,
                    dim=-1
                )

                top_k_probs = torch.softmax(
                    top_k_logits,
                    dim=-1
                )

                sampled_index = torch.multinomial(
                    top_k_probs,
                    num_samples=1
                )

                next_token = torch.gather(
                    top_k_indices,
                    -1,
                    sampled_index
                )

            # Top-p sampling
            elif strategy == "top_p":

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

                sorted_indices_to_remove = cumulative_probs > top_p

                sorted_indices_to_remove[..., 1:] = (
                    sorted_indices_to_remove[..., :-1].clone()
                )

                sorted_indices_to_remove[..., 0] = False

                sorted_logits[sorted_indices_to_remove] = float("-inf")

                filtered_probs = torch.softmax(
                    sorted_logits,
                    dim=-1
                )

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

                raise ValueError(
                    f"Unknown generation strategy: {strategy}"
                )

            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

            if next_token.item() == tokenizer.token_to_id("[EOS]"):
                break

    generated_text = tokenizer.decode(
        idx[0].tolist(),
        skip_special_tokens=True
    )

    return generated_text



def load_expanded_model():
    model = GPT().to(DEVICE)

    checkpoint = torch.load(
        "checkpoints_expanded/best_model.pth",
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


if __name__ == "__main__":

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    original_model = load_model()
    expanded_model = load_expanded_model()

    prompt = "नेपालको इतिहास"

    for model_name, model in [
        ("Original model", original_model),
        ("Expanded model", expanded_model)
    ]:

        output = generate_text(
            model,
            tokenizer,
            prompt,
            strategy="top_p",
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2
        )

        print(f"\\n{model_name}:")
        print(output)
