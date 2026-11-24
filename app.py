import streamlit as st
from loguru import logger
from core.graph import app_graph, State
from core.ui import render_sources, privacy_note
from core.llm.registry import (
    build_llm,
    fetch_available_groq_models,
    fetch_available_openai_models,
    fetch_available_gemini_models,
    fetch_available_anthropic_models
)
from core.rerank import maybe_rerank
from core.mcp.adapters import MCPTools
from core.synthesize import _messages

st.set_page_config(page_title="OpenScout — Perplexity-Style RAG", layout="wide")

st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"] {font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;}
        body {padding-top:1.25rem;}
        .block-container{padding:1.5rem 2rem;}
        h1{font-size:32px !important; margin-bottom: 0.35rem;}
        h2{font-size:18px !important;}
        p, .stMarkdown, .stText, .stCaption {font-size:14px !important;}
        .stButton>button {padding:6px 10px !important;}
        section[data-testid="stSidebar"] .block-container {padding:0.75rem 0.85rem;}
        .openscout-source {font-size:13px; color:#444444;}
        .openscout-source-url {font-size:12px; color:#0a66c2;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "keys" not in st.session_state:
    st.session_state["keys"] = {}
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "last_hits" not in st.session_state:
    st.session_state["last_hits"] = []
if "last_citations" not in st.session_state:
    st.session_state["last_citations"] = []

@st.cache_data(ttl=1800, show_spinner=False)
def _cached_groq_models(api_key: str):
    return fetch_available_groq_models(api_key)

@st.cache_data(ttl=1800, show_spinner=False)
def _cached_openai_models(api_key: str):
    return fetch_available_openai_models(api_key)

@st.cache_data(ttl=1800, show_spinner=False)
def _cached_gemini_models(api_key: str):
    return fetch_available_gemini_models(api_key)

@st.cache_data(ttl=1800, show_spinner=False)
def _cached_anthropic_models(api_key: str):
    return fetch_available_anthropic_models(api_key)

with st.sidebar:
    st.header("OpenScout Settings")

    def key_input(label, name, ph, help_text=None):
        cur = st.session_state["keys"].get(name, "")
        v = st.text_input(label, value=cur, type="password", placeholder=ph, key=f"widget_{name}", help=help_text)
        cleaned = v.strip().strip("'\"")
        st.session_state["keys"][name] = cleaned
        return cleaned

    st.subheader("1. Web Search Key")
    tavily_key = key_input("Tavily API Key", "TAVILY_API_KEY", "tvly-...")
    if not tavily_key:
        st.caption("🔴 Tavily key is required for web search.")
    else:
        st.caption("🟢 Tavily key entered.")

    st.subheader("2. LLM Provider & Model")
    provider = st.selectbox("Provider", ["Groq", "OpenAI", "Gemini", "Anthropic"])
    selected_model = ""

    if provider == "Groq":
        k_val = key_input("Groq API Key", "GROQ_API_KEY", "gsk_...")
        if not k_val:
            st.caption("🔴 Groq key is required.")
            models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        else:
            st.caption("🟢 Groq key entered.")
            models = _cached_groq_models(k_val)

        choice = st.selectbox("Groq Model", models + ["(Custom Model...)"])
        if choice == "(Custom Model...)":
            selected_model = st.text_input("Enter Groq Model ID", value="llama-3.3-70b-versatile", key="custom_groq").strip()
        else:
            selected_model = choice

    elif provider == "OpenAI":
        k_val = key_input("OpenAI API Key", "OPENAI_API_KEY", "sk-...")
        if not k_val:
            st.caption("🔴 OpenAI key is required.")
            models = ["gpt-4o-mini", "gpt-4o", "o1-mini", "o3-mini"]
        else:
            st.caption("🟢 OpenAI key entered.")
            models = _cached_openai_models(k_val)

        choice = st.selectbox("OpenAI Model", models + ["(Custom Model...)"])
        if choice == "(Custom Model...)":
            selected_model = st.text_input("Enter OpenAI Model ID", value="gpt-4o-mini", key="custom_openai").strip()
        else:
            selected_model = choice

    elif provider == "Gemini":
        k_val = key_input("Google API Key", "GOOGLE_API_KEY", "AIza...")
        if not k_val:
            st.caption("🔴 Google Gemini key is required.")
            models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"]
        else:
            st.caption("🟢 Google Gemini key entered.")
            models = _cached_gemini_models(k_val)

        choice = st.selectbox("Gemini Model", models + ["(Custom Model...)"])
        if choice == "(Custom Model...)":
            selected_model = st.text_input("Enter Gemini Model ID", value="gemini-1.5-pro", key="custom_gemini").strip()
        else:
            selected_model = choice

    elif provider == "Anthropic":
        k_val = key_input("Anthropic API Key", "ANTHROPIC_API_KEY", "sk-ant-...")
        if not k_val:
            st.caption("🔴 Anthropic key is required.")
            models = ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus-latest"]
        else:
            st.caption("🟢 Anthropic key entered.")
            models = _cached_anthropic_models(k_val)

        choice = st.selectbox("Anthropic Model", models + ["(Custom Model...)"])
        if choice == "(Custom Model...)":
            selected_model = st.text_input("Enter Anthropic Model ID", value="claude-3-5-sonnet-latest", key="custom_anthropic").strip()
        else:
            selected_model = choice

    with st.expander("Embeddings & MCP Settings"):
        if provider != "OpenAI":
            key_input("OpenAI Key for Embeddings (optional)", "OPENAI_API_KEY", "sk-...")
            st.caption("Leave blank to automatically use local Sentence-Transformers (free, runs locally).")
        mcp_checkbox = st.checkbox("Use MCP tools (Tavily / Neo4j)", value=False)
        if mcp_checkbox:
            key_input("MCP Server URL", "MCP_URL", "http://localhost:8000")
            key_input("MCP API Key (optional)", "MCP_API_KEY", "")
            key_input("Neo4j URI", "NEO4J_URI", "bolt://localhost:7687")
            key_input("Neo4j Username", "NEO4J_USERNAME", "neo4j")
            key_input("Neo4j Password", "NEO4J_PASSWORD", "")

    st.subheader("3. RAG Controls")
    answer_mode = st.selectbox("Answer Mode", ["concise", "detailed", "list", "pros_cons", "timeline", "table"])
    k = st.slider("Sources (Top-k)", min_value=2, max_value=12, value=6)
    use_reranker = st.checkbox("Cross-Encoder Reranker", value=True)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    if st.button("Reset Session & Keys"):
        st.session_state.clear()
        st.rerun()

    privacy_note()

    st.markdown("---")
    st.subheader("Connection Tests")

    col_tav, col_llm = st.columns(2)
    with col_tav:
        if st.button("Test Tavily"):
            current_tav = st.session_state["keys"].get("TAVILY_API_KEY", "").strip()
            if not current_tav:
                st.error("Enter Tavily key first.")
            else:
                try:
                    from tavily import TavilyClient
                    client = TavilyClient(api_key=current_tav)
                    client.search(query="test", max_results=1)
                    st.success("Tavily OK!")
                except Exception as e:
                    st.error(f"Tavily failed: {e}")

    with col_llm:
        if st.button(f"Test {provider}"):
            req_key = "GROQ_API_KEY" if provider == "Groq" else (
                "OPENAI_API_KEY" if provider == "OpenAI" else (
                    "ANTHROPIC_API_KEY" if provider == "Anthropic" else "GOOGLE_API_KEY"
                )
            )
            key_val = st.session_state["keys"].get(req_key, "").strip()
            if not key_val:
                st.error(f"Enter {provider} key first.")
            else:
                try:
                    llm = build_llm(provider, st.session_state["keys"], model=selected_model)
                    msg = [{"role": "user", "content": "Respond with OK."}]
                    resp = llm.chat(msg, stream=False, max_tokens=16)
                    out = ''.join(resp) if hasattr(resp, '__iter__') and not isinstance(resp, str) else str(resp)
                    if out:
                        st.success(f"{provider} ({selected_model}) OK: {out.strip()[:30]}")
                    else:
                        st.warning("Empty response.")
                except Exception as e:
                    st.error(f"{provider} failed: {e}")

st.markdown("<h1 style='text-align: center;'>OpenScout</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Perplexity-style RAG search engine with web retrieval, reranking & inline citations</p>", unsafe_allow_html=True)

left_col, right_col = st.columns([3, 2])

with left_col:
    for m in st.session_state["messages"]:
        if m.get("role") == "user":
            with st.chat_message("user"):
                st.write(m.get("content", ""))
        else:
            with st.chat_message("assistant"):
                st.markdown("**OpenScout**")
                st.write(m.get("content", ""))

with right_col:
    st.markdown("### Sources")
    if st.session_state["last_hits"]:
        render_sources(st.session_state["last_hits"], st.session_state["last_citations"])
    else:
        st.caption("Retrieved and cited sources will appear here.")

query = st.chat_input("Ask anything (we'll search the web, fetch pages, and cite sources)...")

if query:
    active_tavily = st.session_state["keys"].get("TAVILY_API_KEY", "").strip()
    if not active_tavily:
        st.error("Please enter your Tavily API Key in the sidebar before searching.")
        st.stop()

    required_llm_key = "GROQ_API_KEY" if provider == "Groq" else (
        "OPENAI_API_KEY" if provider == "OpenAI" else (
            "ANTHROPIC_API_KEY" if provider == "Anthropic" else "GOOGLE_API_KEY"
        )
    )
    if not st.session_state["keys"].get(required_llm_key, "").strip():
        st.error(f"Please enter your {provider} API Key in the sidebar before searching.")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": query})
    with left_col:
        with st.chat_message("user"):
            st.write(query)

    with left_col:
        with st.chat_message("assistant"):
            st.markdown("**OpenScout**")
            status_container = st.empty()
            status_container.info("Searching web, reading pages, reranking, and synthesizing...")

            try:
                llm = build_llm(provider, st.session_state["keys"], model=selected_model)
                tools = MCPTools(st.session_state["keys"])
                use_mcp = bool(st.session_state["keys"].get("USE_MCP", False) or mcp_checkbox)

                result = app_graph.invoke(State(
                    query=query,
                    k=k,
                    use_mcp=use_mcp,
                    tavily_api_key=active_tavily,
                    openai_api_key=st.session_state["keys"].get("OPENAI_API_KEY", ""),
                    tools=tools,
                    llm=llm
                ))

                if isinstance(result, dict):
                    hits = result.get("hits", []) or []
                else:
                    hits = getattr(result, "hits", []) or []

                if use_reranker:
                    hits = maybe_rerank(query, hits, top_k=k)

                status_container.empty()

                messages_for_llm = _messages(
                    query, hits, mode=answer_mode,
                    history=st.session_state["messages"][:-1]
                )

                answer_placeholder = st.empty()
                full_answer = ""

                try:
                    stream_obj = llm.chat(messages_for_llm, stream=True, temperature=temperature, max_tokens=1024)
                    if hasattr(stream_obj, "__iter__") and not isinstance(stream_obj, str):
                        for chunk in stream_obj:
                            full_answer += chunk if isinstance(chunk, str) else str(chunk)
                            answer_placeholder.markdown(full_answer)
                    else:
                        full_answer = str(stream_obj)
                        answer_placeholder.markdown(full_answer)
                except Exception:
                    full_answer, _ = State.synthesizer(
                        llm, query, hits, mode=answer_mode, temperature=temperature, max_tokens=1024,
                        history=st.session_state["messages"][:-1]
                    )
                    answer_placeholder.markdown(full_answer)

                citations = [{"id": i} for i, _ in enumerate(hits, start=1) if f"[#{i}]" in full_answer]
                st.session_state["messages"].append({"role": "assistant", "content": full_answer})
                st.session_state["last_hits"] = hits
                st.session_state["last_citations"] = citations

                st.rerun()

            except Exception as e:
                status_container.empty()
                st.error("An error occurred during pipeline execution.")
                with st.expander("Error details"):
                    st.write(str(e))
                logger.exception("Error running graph pipeline")
