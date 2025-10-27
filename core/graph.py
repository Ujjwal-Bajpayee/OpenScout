from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from typing import List, Dict, Callable
from .chunk import chunk_text
from .embed import embed_texts_openai, embed_one_openai
from .faiss_store import add_vectors, save_index, search, get_index_and_db
from .fetch import fetch_many
from .search import tavily_search
from .synthesize import synthesize_with_llm

class State(BaseModel):
    query: str
    k: int = 6
    use_mcp: bool = False
    tavily_api_key: str = ""
    openai_api_key: str = ""
    urls: List[str] = []
    pages: List[Dict] = []
    chunks: List[Dict] = []
    hits: List[Dict] = []
    tools: object | None = None  # MCPTools
    synthesizer: Callable = None

def node_search(s: State) -> State:
    if s.use_mcp and s.tools:
        results = s.tools.search(s.query, k=8)
    else:
        results = tavily_search(s.query, s.tavily_api_key, k=8)
    s.urls = [r["url"] for r in results if r.get("url")]
    s._results = results
    return s

def node_fetch(s: State) -> State:
    import asyncio
    s.pages = asyncio.run(fetch_many(s.urls))
    return s

def node_index(s: State) -> State:
    index, conn = get_index_and_db()
    to_chunks = []
    for p in s.pages:
        if not p.get("text"): continue
        for ord_, t in enumerate(chunk_text(p["text"])):
            to_chunks.append({"url": p["url"], "title": p["title"], "ord": ord_, "text": t, "domain": p["domain"]})
    if not to_chunks:
        s.chunks = []
        return s
    # prefer API key from State (passed from app session) otherwise fall back to env
    X = embed_texts_openai([c["text"] for c in to_chunks], api_key=s.openai_api_key)
    add_vectors(index, conn, X, to_chunks)
    save_index(index)
    s.chunks = to_chunks
    return s

def node_retrieve(s: State) -> State:
    index, conn = get_index_and_db()
    qvec = embed_one_openai(s.query, api_key=s.openai_api_key)
    s.hits = search(index, conn, qvec, k=s.k)
    return s

g = StateGraph(State)
g.add_node("search", node_search)
g.add_node("fetch", node_fetch)
g.add_node("index", node_index)
g.add_node("retrieve", node_retrieve)
g.set_entry_point("search")
g.add_edge("search","fetch")
g.add_edge("fetch","index")
g.add_edge("index","retrieve")
app_graph = g.compile()

def synthesizer(llm, query: str, hits: List[Dict], mode: str, temperature: float, max_tokens: int):
    return synthesize_with_llm(llm, query, hits, mode, temperature, max_tokens)

State.synthesizer = staticmethod(synthesizer)
