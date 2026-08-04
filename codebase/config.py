"""
Configuration for the RAG app.

Centralizes the model endpoint so it can point at either:
  - a local Granite instance (default, air-gapped/offline mode), or
  - an internal granite-deployed model endpoint on OpenShift AI.

Values can be overridden via environment variables without editing this file.
"""

import os

# ---------------------------------------------------------------------------
# Chat / vision model endpoint (Granite Vision)
# ---------------------------------------------------------------------------
MODEL_BASE_URL = os.environ.get(
    "MODEL_BASE_URL",
    "http://granite-vision-model-predictor.user9.svc.cluster.local",
)
MODEL_NAME = os.environ.get("MODEL_NAME", "granite-vision-model")
MODEL_API_KEY = os.environ.get("MODEL_API_KEY", "not-needed")
MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "openai")

# ---------------------------------------------------------------------------
# Generation parameters (chat model)
# ---------------------------------------------------------------------------
MODEL_TEMPERATURE = float(os.environ.get("MODEL_TEMPERATURE", "0.1"))
MODEL_MAX_TOKENS = int(os.environ.get("MODEL_MAX_TOKENS", "128"))

# ---------------------------------------------------------------------------
# Embedding model endpoint (all-MiniLM-L6-v2)
# ---------------------------------------------------------------------------
EMBEDDING_BASE_URL = os.environ.get(
    "EMBEDDING_BASE_URL",
    "http://redhataiall-minilm-l6-v2-predictor.user9.svc.cluster.local",
)
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "redhataiall-minilm-l6-v2")
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "not-needed")
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "openai")