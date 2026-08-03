"""
Configuration for the RAG app.

Centralizes the model endpoint so it can point at either:
  - a local Ollama instance (default, air-gapped/offline mode), or
  - an internal NIM-deployed model endpoint on OpenShift AI.

Values can be overridden via environment variables without editing this file.
"""

import os

# ---------------------------------------------------------------------------
# Model endpoint
# ---------------------------------------------------------------------------

MODEL_BASE_URL = os.environ.get(
    "MODEL_BASE_URL",
    "http://nim-gpt-oss-20b-predictor.pw-demo.svc.cluster.local",
)

MODEL_NAME = os.environ.get("MODEL_NAME", "openai/gpt-oss-20b")

MODEL_API_KEY = os.environ.get("MODEL_API_KEY", "not-needed")

MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "openai")

# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------

MODEL_TEMPERATURE = float(os.environ.get("MODEL_TEMPERATURE", "0.1"))
MODEL_MAX_TOKENS = int(os.environ.get("MODEL_MAX_TOKENS", "128"))