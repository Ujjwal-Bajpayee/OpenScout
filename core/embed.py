import os
import numpy as np
from openai import OpenAI
from .faiss_store import l2_normalize


def _client(api_key: str | None):
    # allow explicit api_key or fall back to env var
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        # Provide a clearer error message than the OpenAI client so the UI can show guidance
        raise RuntimeError(
            "OpenAI API key not provided. Set OPENAI_API_KEY in .env or paste it in the sidebar."
        )
    return OpenAI(api_key=key)


def embed_texts_openai(texts, api_key: str | None = None, model="text-embedding-3-small"):
    res = _client(api_key).embeddings.create(model=model, input=texts)
    X = np.array([d.embedding for d in res.data], dtype=np.float32)
    return l2_normalize(X)


def embed_one_openai(text, api_key: str | None = None, model="text-embedding-3-small"):
    return embed_texts_openai([text], api_key, model=model)[0]
