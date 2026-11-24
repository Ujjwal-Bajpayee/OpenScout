import os
import sqlite3
import faiss
import numpy as np
import streamlit as st

@st.cache_resource
def get_index_and_db(dim: int = 1536):
    idx_path = "faiss_index.bin"
    db_path = "chunks.sqlite"
    index = None

    if os.path.exists(idx_path):
        try:
            loaded = faiss.read_index(idx_path)
            if getattr(loaded, "d", None) == dim:
                index = loaded
                if not hasattr(index, "add_with_ids"):
                    index = faiss.IndexIDMap(index)
        except Exception:
            index = None

    if index is None:
        flat = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap(flat)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS chunks(
        id INTEGER PRIMARY KEY, url TEXT, title TEXT, ord INT,
        text TEXT, domain TEXT, embedding_dim INT, created_at TEXT
    );""")
    return index, conn

def save_index(index, path: str = "faiss_index.bin"):
    faiss.write_index(index, path)

def l2_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n

def _max_id(conn):
    row = conn.execute("SELECT MAX(id) FROM chunks").fetchone()
    return row[0] if row and row[0] is not None else -1

def add_vectors(index, conn, X, metas):
    if len(metas) == 0:
        return np.array([], dtype=np.int64)
    start = _max_id(conn) + 1
    ids = (start + np.arange(len(metas))).astype(np.int64)
    index.add_with_ids(X.astype(np.float32), ids)
    with conn:
        for i, m in enumerate(metas):
            conn.execute("""INSERT INTO chunks(id,url,title,ord,text,domain,embedding_dim,created_at)
                            VALUES(?,?,?,?,?,?,?,datetime('now'))""",
                         (int(ids[i]), m.get("url", ""), m.get("title", ""), m.get("ord", 0),
                          m.get("text", ""), m.get("domain", ""), int(X.shape[1])))
    return ids

def fetch_by_ids(conn, ids):
    if not ids:
        return []
    valid_ids = [int(i) for i in ids if i >= 0]
    if not valid_ids:
        return []
    q = ",".join("?" * len(valid_ids))
    rows = conn.execute(f"SELECT id,url,title,ord,text,domain FROM chunks WHERE id IN ({q})", valid_ids).fetchall()
    m = {r[0]: r for r in rows}
    out = []
    for i in valid_ids:
        if i in m:
            id_, url, title, ord_, text, dom = m[i]
            out.append(dict(id=id_, url=url, title=title, ord=ord_, text=text, domain=dom))
    return out

def search(index, conn, qvec, k=6, overfetch=4):
    total = index.ntotal
    if total == 0:
        return []
    q = l2_normalize(qvec.reshape(1, -1)).astype(np.float32)
    fetch_k = min(total, max(k * overfetch, k))
    scores, ids = index.search(q, fetch_k)
    valid_pairs = [(int(id_), float(sc)) for id_, sc in zip(ids[0], scores[0]) if id_ >= 0]
    if not valid_pairs:
        return []
    id_list = [p[0] for p in valid_pairs]
    score_map = {p[0]: p[1] for p in valid_pairs}
    hits = fetch_by_ids(conn, id_list)
    for h in hits:
        h["score"] = score_map.get(h["id"], 0.0)
    return hits[:k]
