from sentence_transformers import CrossEncoder
_model = None
def _get():
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model

def maybe_rerank(query: str, hits: list[dict], top_k: int = 6):
    if not hits: return hits
    model = _get()
    pairs = [(query, h.get("text","")) for h in hits]
    scores = model.predict(pairs).tolist()
    ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [h for h,_ in ranked]
