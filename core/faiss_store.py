import os, sqlite3, faiss, numpy as np, time
import streamlit as st
from .constants import EMBED_DIM

@st.cache_resource
def get_index_and_db():
    idx_path, db_path = "faiss_index.bin", "chunks.sqlite"
    # Use an IndexFlatIP wrapped in an IndexIDMap so we can add vectors with explicit ids.
    # If a saved index exists, load it; if it doesn't support add_with_ids, wrap it.
    index = faiss.IndexFlatIP(EMBED_DIM)
    # wrap to provide add_with_ids / id lookups
    try:
        index = faiss.IndexIDMap(index)
    except Exception:
        pass
    if os.path.exists(idx_path):
        index = faiss.read_index(idx_path)
        # If the loaded index doesn't support add_with_ids, wrap it in an ID map
        if not hasattr(index, "add_with_ids"):
            try:
                index = faiss.IndexIDMap(index)
            except Exception:
                # final fallback: raise a helpful error
                raise RuntimeError("Loaded FAISS index does not support add_with_ids and cannot be wrapped.")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS chunks(
        id INTEGER PRIMARY KEY, url TEXT, title TEXT, ord INT,
        text TEXT, domain TEXT, embedding_dim INT, created_at TEXT
    );""")
    return index, conn

def save_index(index, path="faiss_index.bin"):
    faiss.write_index(index, path)

def l2_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n

def _max_id(conn): row = conn.execute("SELECT MAX(id) FROM chunks").fetchone(); return row[0] if row and row[0] else -1

def add_vectors(index, conn, X, metas):
    start = _max_id(conn) + 1
    ids = (start + np.arange(len(metas))).astype(np.int64)
    index.add_with_ids(X.astype(np.float32), ids)
    with conn:
        for i, m in enumerate(metas):
            conn.execute("""INSERT INTO chunks(id,url,title,ord,text,domain,embedding_dim,created_at)
                            VALUES(?,?,?,?,?,?,?,datetime('now'))""",
                         (int(ids[i]), m.get("url",""), m.get("title",""), m.get("ord",0),
                          m.get("text",""), m.get("domain",""), X.shape[1]))
    return ids

def fetch_by_ids(conn, ids):
    if not ids: return []
    q = ",".join("?"*len(ids))
    rows = conn.execute(f"SELECT id,url,title,ord,text,domain FROM chunks WHERE id IN ({q})", ids).fetchall()
    m = {r[0]: r for r in rows}
    out = []
    for i in ids:
        if i in m:
            id_,url,title,ord_,text,dom = m[i]
            out.append(dict(id=id_, url=url, title=title, ord=ord_, text=text, domain=dom))
    return out

def search(index, conn, qvec, k=6, overfetch=4):
    q = l2_normalize(qvec.reshape(1,-1)).astype(np.float32)
    scores, ids = index.search(q, k*overfetch)
    ids = ids[0].tolist()
    return fetch_by_ids(conn, ids)[:k]
