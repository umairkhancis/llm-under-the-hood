from pathlib import Path
import sys

import tiktoken
import torch

# The finetuned weights always live inside this module's directory, regardless
# of where the app was launched from.
MODULE_DIR = Path(__file__).resolve().parent

from gpt_module.gpt import GPTModel
from embeddings_module.embeddings import text_to_token_ids, token_ids_to_text
from gpt_training_module.text_generation_app import text_generation_app

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_and_tokenizer():
    """
    Code to load a GPT-2 model with finetuned weights generated in chapter 7.
    This requires that you run the code in chapter 7 first, which generates the necessary gpt2-medium355M-sft.pth file.
    """
    GPT_CONFIG_355M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_window_size": 1024,  # Shortened context length (orig: 1024)
        "emb_dim": 1024,         # Embedding dimension
        "n_heads": 16,           # Number of attention heads
        "n_layers": 24,          # Number of layers
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    tokenizer = tiktoken.get_encoding("gpt2")

    model_path = MODULE_DIR / "finetuned-weights" / "gpt2-medium355M-sft.pth"
    print(model_path)

    if not model_path.exists():
        print(
            f"Could not find the {model_path} file. Please run the chapter 7 code "
            " (ch07.ipynb) to generate the gpt2-medium355M-sft.pt file."
        )
        sys.exit()
    checkpoint = torch.load(model_path, weights_only=True)

    model = GPTModel(GPT_CONFIG_355M)
    model.load_state_dict(checkpoint)
    model.to(device)

    return tokenizer, model, GPT_CONFIG_355M

def extract_response(response_text, input_text):
    return response_text[len(input_text):].replace("### Response:", "").strip()

# Load once at import time so the app starts with the model ready, rather than
# reloading the weights on every message.
tokenizer, model, model_config = get_model_and_tokenizer()

def generate_response(instruction):
    torch.manual_seed(123)

    prompt = f"""Below is an instruction that describes a task. Write a response
    that appropriately completes the request.

    ### Instruction:
    {instruction}
    """

    token_ids = text_generation_app(
        model=model,
        input_text=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=35,
        context_window_size=model_config["context_window_size"],
        eos_id=50256
    )

    text = token_ids_to_text(token_ids, tokenizer)
    return extract_response(text, prompt)
