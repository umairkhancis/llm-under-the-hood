# LLM Under the Hood

A hands-on journey toward *tafahhum* (تفهم) — a deep, earned understanding of Large Language Models (LLMs), by building GPT (Generative Pretrained Transformer), which is based on the decoder-only transformer architecture, from scratch in PyTorch.

> _After reading these computational essays, the terms used in the above sentence will not remain obscured._

Nothing here is a black box. 

The repo follows one continuous path, each stage built on the previous: 
1. turning text into **token embeddings**,
2. letting tokens see each other through **self-attention**,
3. widening that view with **causal multi-head attention**,
4. stacking it all into the **transformer blocks** of a full GPT model,
5. **pretraining** it on raw text (then loading OpenAI's GPT-2 weights),
6. **fine-tuning it to follow instructions**, and
7. finally serving the finetuned model behind a **chat UI** with Chainlit.

Every stage is a runnable computational essay paired with the module code it explains.

## `Umair-GPT` behind UI Application

<img width="1507" height="857" alt="Screenshot 2026-07-27 at 6 03 35 PM" src="https://github.com/user-attachments/assets/0592b9fb-77da-4ccf-b650-8f4ca8157e3e" />

## Video Demo

https://github.com/user-attachments/assets/1ec30eb7-8809-4905-8c97-bd411d9009a2

## Suggested learning path

Follow the computational essays in this order — each stage builds directly on the previous one:

| # | Topic | Learning outcome |
|---|-------|------------------|
| 1 | [Tokenization & embeddings](https://github.com/umairkhancis/llm-under-the-hood/blob/main/embeddings_module/notebooks/tokenization-embedding-essay.ipynb) | Understand how raw text becomes numbers — split into tokens and mapped to trainable embedding vectors that capture meaning and position. |
| 2 | [Simple self-attention](https://github.com/umairkhancis/llm-under-the-hood/blob/main/attention_mechanism_module/notebooks/essay.ipynb) | See why attention replaced RNNs and compute the simplest form of it: dot-product similarity between raw embeddings turned into context vectors. |
| 3 | [Trainable attention weights](https://github.com/umairkhancis/llm-under-the-hood/blob/main/attention_mechanism_module/notebooks/trainable-attention-essay.ipynb) | Introduce the learnable query, key, and value matrices so the model can learn *how* words should attend to each other, not just how similar they are. |
| 4 | [Causal (masked) attention](https://github.com/umairkhancis/llm-under-the-hood/blob/main/attention_mechanism_module/notebooks/causal-attention-essay.ipynb) | Mask out future tokens so each position only attends to the past — the property that lets a decoder-only model predict the next token. |
| 5 | [Multi-head attention](https://github.com/umairkhancis/llm-under-the-hood/blob/main/attention_mechanism_module/notebooks/multihead-attention-essay.ipynb) | Run several attention heads in parallel so each can specialize in different linguistic patterns, yielding richer contextual representations. |
| 6 | [The transformer block](https://github.com/umairkhancis/llm-under-the-hood/blob/main/gpt_module/notebooks/transformer-block-essay.ipynb) | Wrap attention with layer normalization, feed-forward layers, and residual (shortcut) connections into the repeatable block transformers are made of. |
| 7 | [The full GPT model](https://github.com/umairkhancis/llm-under-the-hood/blob/main/gpt_module/notebooks/gpt-model-essay.ipynb) | Stack embeddings and transformer blocks into a complete `GPTModel` and generate text autoregressively, one next token at a time. |
| 8 | [Pretraining the model](https://github.com/umairkhancis/llm-under-the-hood/blob/main/gpt_training_module/notebooks/gpt-training-essay.ipynb) | Quantify output quality with a cross-entropy loss and write the training loop that pretrains the model on raw text until it produces coherent language. |
| 9 | [Text decoding strategies (temperature, top-k)](https://github.com/umairkhancis/llm-under-the-hood/blob/main/gpt_training_module/notebooks/text-decoding-essay.ipynb) | Turn raw logits into "creative" text — control randomness and diversity of generation with temperature scaling and top-k sampling. |
| 10 | [Loading OpenAI's GPT-2 open weights](https://github.com/umairkhancis/llm-under-the-hood/blob/main/gpt_training_module/notebooks/open-weights-essay.ipynb) | Load OpenAI's pretrained GPT-2 weights into our own implementation — leveraging their training budget instead of paying for pretraining ourselves. |
| 11 | [Fine-tuning (classification & instruction)](https://github.com/umairkhancis/llm-under-the-hood/blob/main/finetuning-module/notebooks/finetuning-essay.ipynb) | Fine-tune the pretrained model on labeled data — turning a next-token predictor into a classifier and an instruction-following assistant. |
| 12 | [Serving the model behind a chat UI](https://github.com/umairkhancis/llm-under-the-hood/blob/main/app.py) | Serve the finetuned model behind a Chainlit chat interface (`chainlit run app.py`) — from tensors to a working AI assistant. |

## Setup

```bash
git clone git@github.com:umairkhancis/llm-under-the-hood.git
cd llm-under-the-hood

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Read & Practice with Computational Essays:
```
jupyter lab
```

Run the Final Chainlit UI App:
```
chainlit run app.py --headless
```

### Run the Chainlit app with Docker

The app can also run in a container. The image bundles only the runtime
dependencies (CPU-only PyTorch, tiktoken, Chainlit) and runs as a non-root
user; the finetuned weights are not baked in — they are bind-mounted
read-only at runtime.

Requires `inference_module/finetuned-weights/gpt2-medium355M-sft.pth` to
exist locally (generated by the fine-tuning essay, stage 11).

```bash
docker build -t llm-under-the-hood .

docker run --rm -d -p 8000:8000 \
  -v "$PWD/inference_module/finetuned-weights:/home/appuser/app/inference_module/finetuned-weights:ro" \
  --name gpt-chat llm-under-the-hood
```

Then open http://localhost:8000. Stop it with `docker stop gpt-chat`.

Key dependencies: **PyTorch** (the model), **tiktoken** (GPT-2's BPE tokenizer), **TensorFlow** (only to read OpenAI's original checkpoint files), **matplotlib/pandas** (plots and data wrangling).

### A note on imports

The computational essays import earlier modules by adding the repo root to the path:

```python
import sys
sys.path.append("../..")   # from a notebooks/ folder up to the repo root

from attention_mechanism_module import MultiHeadAttention
from transformer_module import TransformerBlock
from gpt_module import GPTModel
```

Run each computational essay from its own directory (the default in Jupyter) so the relative paths to `data/` and `images/` resolve.

### Large files are not in the repo

Trained checkpoints (`*.pth`) and downloaded GPT-2 weights (`gpt2/` folders) are gitignored — they are hundreds of MB to GBs. The notebooks regenerate or re-download everything they need; GPT-2 weights are fetched with `gpt_training_module/gpt_download.py`.

## Acknowledgements

- Sebastian Raschka's [*Build a Large Language Model From Scratch*](https://www.manning.com/books/build-a-large-language-model-from-scratch) — the primary guide for this journey. `gpt_download.py` is from the book's [official repository](https://github.com/rasbt/LLMs-from-scratch) (Apache 2.0).
- OpenAI's [GPT-2](https://github.com/openai/gpt-2) — the open weights loaded and fine-tuned here.
