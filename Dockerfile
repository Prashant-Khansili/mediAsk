FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as UID 1000, so the app must live in a
# directory that user can write to (uploads and index rebuilds happen at runtime).
RUN useradd -m -u 1000 user && \
    mkdir -p /home/user/app && \
    chown -R user:user /home/user

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

# The extra index provides the CPU-only torch wheel, which is far smaller than
# the default CUDA build and is all a free CPU host can use.
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Bake the embedding model into the image so the first question is not slow.
# It must match "embedding_model" in rag/faiss_index/index_config.json.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-small-v2')"

COPY --chown=user:user . .
RUN mkdir -p data

# 7860: Streamlit UI (the port Hugging Face Spaces serves). 8000: FastAPI.
EXPOSE 7860 8000

# Run the API in the background and the UI in the foreground. XSRF protection
# is disabled because the Spaces iframe otherwise blocks Streamlit file uploads.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & exec streamlit run app/ui.py --server.port 7860 --server.address 0.0.0.0 --server.headless true --server.enableXsrfProtection false"]
