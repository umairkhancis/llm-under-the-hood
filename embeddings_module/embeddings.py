import torch

# Helper function to convert python list to tensors after invoking tokenizer to encode text into token_ids.
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

# Helper function to convert tensors back to python list and invoke tokenizer decode token_ids to text.
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())