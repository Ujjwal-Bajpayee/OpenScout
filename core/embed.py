import os
import numpy as np

_local_embed_model = None

def _get_local_model():
    global _local_embed_model
    if _local_embed_model is None:
        from sentence_transformers import SentenceTransformer
        _local_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_embed_model

def l2_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n

def embed_texts(texts: list[str], api_key: str | None = None, model: str = "text-embedding-3-small") -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if key:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        res = client.embeddings.create(model=model, input=texts)
        X = np.array([d.embedding for d in res.data], dtype=np.float32)
        return l2_normalize(X)

    st_model = _get_local_model()
    embeddings = st_model.encode(texts, convert_to_numpy=True)
    return l2_normalize(embeddings.astype(np.float32))

def embed_one(text: str, api_key: str | None = None, model: str = "text-embedding-3-small") -> np.ndarray:
    return embed_texts([text], api_key=api_key, model=model)[0]

embed_texts_openai = embed_texts
embed_one_openai = embed_one
