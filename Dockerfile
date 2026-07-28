FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# CPU-only torch index keeps the image far smaller than the default CUDA build
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.2.2" && \
    pip install --no-cache-dir "tiktoken>=0.5.1" "chainlit>=1.0.0"

RUN useradd --create-home appuser
WORKDIR /home/appuser/app

# Only the modules the app imports; code stays root-owned so the runtime user
# cannot modify it
COPY app.py ./
COPY inference_module/inference.py inference_module/
COPY gpt_module/ gpt_module/
COPY embeddings_module/ embeddings_module/
# Deliberately without the package __init__.py: it eagerly imports training
# utilities that need tensorflow/numpy/matplotlib, none of which the app uses
COPY gpt_training_module/text_generation_app.py gpt_training_module/
COPY transformer_module/ transformer_module/
COPY attention_mechanism_module/ attention_mechanism_module/

# The only paths chainlit writes to at runtime
RUN mkdir .chainlit .files && touch chainlit.md && \
    chown appuser .chainlit .files chainlit.md

USER appuser
EXPOSE 8000

# Weights are mounted read-only at runtime, not baked into the image:
#   docker run --rm -p 8000:8000 \
#     -v "$PWD/inference_module/finetuned-weights:/home/appuser/app/inference_module/finetuned-weights:ro" \
#     llm-under-the-hood
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000", "--headless"]
