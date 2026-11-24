import warnings
warnings.filterwarnings("ignore")
import asyncio
import concurrent.futures
from typing import List, Dict, Callable
from pydantic import BaseModel
from langgraph.graph import StateGraph
from .chunk import chunk_text
from .embed import embed_texts, embed_one
from .faiss_store import add_vectors, save_index, search, get_index_and_db
from .fetch import fetch_many
from .search import tavily_search
from .synthesize import synthesize_with_llm

class State(BaseModel):
    query: str
    optimized_query: str = ""
    k: int = 6
    use_mcp: bool = False
    tavily_api_key: str = ""
    openai_api_key: str = ""
    urls: List[str] = []
    search_results: List[Dict] = []
    pages: List[Dict] = []
    chunks: List[Dict] = []
    hits: List[Dict] = []
    tools: object | None = None
    llm: object | None = None
    synthesizer: Callable = None

def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return executor.submit(lambda: asyncio.run(coro)).result()
    return asyncio.run(coro)

def _optimize_query(llm, user_query: str) -> str:
    if not llm:
        return user_query
    try:
        msgs = [
            {"role": "system", "content": "You are a search query optimizer. Given a user question, generate a single concise keyword search query targeting specific facts, entities, names, records, or numbers. Output ONLY the search query text, nothing else."},
            {"role": "user", "content": user_query}
        ]
        resp = llm.chat(msgs, stream=False, max_tokens=32, temperature=0.0)
        out = ''.join(resp) if hasattr(resp, '__iter__') and not isinstance(resp, str) else str(resp)
        cleaned = out.strip().strip('"\'')
        return cleaned if cleaned and len(cleaned) > 2 else user_query
    except Exception:
        return user_query

def node_search(s: State) -> State:
    target_query = _optimize_query(s.llm, s.query)
    s.optimized_query = target_query
    if s.use_mcp and s.tools:
        results = s.tools.search(target_query, k=s.k)
    else:
        results = tavily_search(target_query, s.tavily_api_key, k=s.k)
    s.search_results = results
    s.urls = [r["url"] for r in results if r.get("url")]
    return s

def node_fetch(s: State) -> State:
    items = s.search_results if s.search_results else s.urls
    s.pages = _run_async(fetch_many(items))
    return s

def node_index(s: State) -> State:
    to_chunks = []
    for p in s.pages:
        text = p.get("text", "")
        if not text:
            continue
        for ord_, t in enumerate(chunk_text(text)):
            to_chunks.append({
                "url": p.get("url", ""),
                "title": p.get("title", ""),
                "ord": ord_,
                "text": t,
                "domain": p.get("domain", "")
            })

    if not to_chunks:
        s.chunks = []
        return s

    X = embed_texts([c["text"] for c in to_chunks], api_key=s.openai_api_key)
    dim = X.shape[1] if len(X.shape) > 1 and X.shape[1] > 0 else 1536
    index, conn = get_index_and_db(dim=dim)
    add_vectors(index, conn, X, to_chunks)
    save_index(index)
    s.chunks = to_chunks
    return s

def node_retrieve(s: State) -> State:
    retrieve_q = s.optimized_query if s.optimized_query else s.query
    qvec = embed_one(retrieve_q, api_key=s.openai_api_key)
    dim = qvec.shape[0] if len(qvec.shape) > 0 else 1536
    index, conn = get_index_and_db(dim=dim)
    s.hits = search(index, conn, qvec, k=s.k)
    return s

g = StateGraph(State)
g.add_node("search", node_search)
g.add_node("fetch", node_fetch)
g.add_node("index", node_index)
g.add_node("retrieve", node_retrieve)
g.set_entry_point("search")
g.add_edge("search", "fetch")
g.add_edge("fetch", "index")
g.add_edge("index", "retrieve")
app_graph = g.compile()

def synthesizer(llm, query: str, hits: List[Dict], mode: str, temperature: float, max_tokens: int, history: List[Dict] = None):
    return synthesize_with_llm(llm, query, hits, mode, temperature, max_tokens, history=history)

State.synthesizer = staticmethod(synthesizer)
