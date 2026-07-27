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

## Suggested learning path

Follow the computational essays in this order — each stage builds directly on the previous one:

| # | Topic | Computational essay |
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
| 12 | Serving the model behind a chat UI | `inference_module/inference.py` (`chainlit run inference_module/inference.py`) |

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
chainlit run inference_module/inference.py --headless
```

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
