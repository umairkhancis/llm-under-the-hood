# LLM Under the Hood

Building a GPT-style large language model **from scratch, in PyTorch** — to understand exactly what happens under the hood: how text becomes tokens, how attention works, how a transformer block is wired, how the model is trained, and how pretrained open weights are loaded and fine-tuned.

The material follows the spirit of Sebastian Raschka's book [*Build a Large Language Model From Scratch*](https://github.com/rasbt/LLMs-from-scratch), but is organized as a series of **essay notebooks** — each one a narrated, runnable walkthrough of a single concept — plus clean, reusable Python modules extracted from them.

## How this repo is organized

Each stage of the LLM pipeline lives in its own module. Inside each module:

- `notebooks/` — the **essay notebooks**: read these top-to-bottom, run every cell. They contain the explanations, diagrams, and experiments.
- `*.py` files at the module root — the **distilled code**: once a concept is worked out in a notebook, the final classes/functions are extracted here so later modules can import them (e.g. the GPT model imports `TransformerBlock`, which imports `MultiHeadAttention`).

```
embeddings_module/          Tokenization, BPE, token & positional embeddings
attention_mechanism_module/ Self-attention → causal attention → multi-head attention
transformer_module/         TransformerBlock: LayerNorm, GELU, FeedForward, shortcuts
gpt_module/                 The full GPTModel assembled from the pieces above
data_loader_module/         Sliding-window dataset & dataloader for next-token training
gpt_training_module/        Pretraining loop, loss, text decoding, loading GPT-2 weights
finetuning-module/          Classification & instruction fine-tuning of the pretrained model
```

## Suggested learning path

Follow the notebooks in this order — each stage builds directly on the previous one:

| # | Topic | Notebook |
|---|-------|----------|
| 1 | Tokenization & embeddings | `embeddings_module/notebooks/tokenization-embedding-essay.ipynb` |
| 2 | Simple self-attention | `attention_mechanism_module/notebooks/essay.ipynb` |
| 3 | Trainable attention weights | `attention_mechanism_module/notebooks/trainable-attention-essay.ipynb` |
| 4 | Causal (masked) attention | `attention_mechanism_module/notebooks/causal-attention-essay.ipynb` |
| 5 | Multi-head attention | `attention_mechanism_module/notebooks/multihead-attention-essay.ipynb` |
| 6 | The transformer block | `gpt_module/notebooks/transformer-block-essay.ipynb` |
| 7 | The full GPT model | `gpt_module/notebooks/gpt-model-essay.ipynb` |
| 8 | Pretraining the model | `gpt_training_module/notebooks/gpt-training-essay.ipynb` |
| 9 | Text decoding strategies (temperature, top-k) | `gpt_training_module/notebooks/text-decoding-essay.ipynb` |
| 10 | Loading OpenAI's GPT-2 open weights | `gpt_training_module/notebooks/open-weights-essay.ipynb` |
| 11 | Fine-tuning (classification & instruction) | `finetuning-module/notebooks/finetuning-essay.ipynb` |

The `*-practice.ipynb` notebooks are exercise/scratch companions to the essays.

## The big picture

By the end of the path you will have built and understood this data flow:

```
text ──tokenizer──▶ token IDs ──tok_emb + pos_emb──▶ embeddings
      ──▶ [ TransformerBlock × N ]           each block:
             LayerNorm → Multi-Head Attention → +shortcut
             LayerNorm → FeedForward (GELU)   → +shortcut
      ──▶ final LayerNorm ──out_head──▶ logits over vocabulary
      ──cross-entropy vs next token──▶ loss ──▶ backprop ──▶ trained weights
```

…and then reused it three ways: trained from scratch on a small corpus, loaded with OpenAI's GPT-2 124M/355M checkpoints, and fine-tuned for spam classification and instruction following.

## Setup

```bash
git clone git@github.com:umairkhancis/llm-under-the-hood.git
cd llm-under-the-hood

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

jupyter lab
```

Key dependencies: **PyTorch** (the model), **tiktoken** (GPT-2's BPE tokenizer), **TensorFlow** (only to read OpenAI's original checkpoint files), **matplotlib/pandas** (plots and data wrangling).

### A note on imports

The notebooks import earlier modules by adding the repo root to the path:

```python
import sys
sys.path.append("../..")   # from a notebooks/ folder up to the repo root

from attention_mechanism_module import MultiHeadAttention
from transformer_module import TransformerBlock
from gpt_module import GPTModel
```

Run notebooks from their own directory (the default in Jupyter) so the relative paths to `data/` and `images/` resolve.

### Large files are not in the repo

Trained checkpoints (`*.pth`) and downloaded GPT-2 weights (`gpt2/` folders) are gitignored — they are hundreds of MB to GBs. The notebooks regenerate or re-download everything they need; GPT-2 weights are fetched with `gpt_training_module/gpt_download.py`.

## Acknowledgements

- Sebastian Raschka's [*Build a Large Language Model From Scratch*](https://www.manning.com/books/build-a-large-language-model-from-scratch) — the primary guide for this journey. `gpt_download.py` is from the book's [official repository](https://github.com/rasbt/LLMs-from-scratch) (Apache 2.0).
- OpenAI's [GPT-2](https://github.com/openai/gpt-2) — the open weights loaded and fine-tuned here.
