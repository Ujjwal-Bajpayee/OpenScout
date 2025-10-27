import streamlit as st
import os
from dotenv import load_dotenv
from loguru import logger
from core.graph import app_graph, State
from core.ui import render_sources, privacy_note
from core.llm.registry import build_llm
from core.faiss_store import get_index_and_db
from core.rerank import maybe_rerank
from core.mcp.adapters import MCPTools

st.set_page_config(page_title="OpenScout — RAG with MCP tool layer", layout="wide")

st.markdown(
        """
        <style>
            /* Base font sizes and container */
            html, body, [data-testid="stAppViewContainer"] {font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;}
            body {padding-top:1.25rem;}
            .block-container{padding:1.5rem 2rem;}
            h1{font-size:40px !important; margin-bottom: 0.35rem;}
            h2{font-size:15px !important;}
            p, .stMarkdown, .stText, .stCaption {font-size:14px !important;}
            .stButton>button {padding:6px 10px !important;}
            section[data-testid="stSidebar"] .block-container {padding:0.75rem 0.85rem;}
            /* Smaller source captions */
            .openscout-source {font-size:13px; color:#444444;}
            .openscout-source-url {font-size:12px; color:#0a66c2;}
            .openscout-title {text-align:center; font-weight:600; font-size:24px; margin:0.35rem 0 0.75rem 0; color:#111}
        </style>
        """,
        unsafe_allow_html=True,
)

with st.sidebar:
    st.header(" API Keys ")
    # Load .env into environment (allows BYO keys in .env)
    load_dotenv()
    env_defaults = {
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", ""),
        "NEO4J_URI": os.getenv("NEO4J_URI", ""),
        "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME", ""),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", ""),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", "")
    }
    if "keys" not in st.session_state:
        # initialize empty session keys dict
        st.session_state["keys"] = {}
    for _k, _v in env_defaults.items():
        if _v and not st.session_state["keys"].get(_k):
            st.session_state["keys"][_k] = _v

    def key(label, name, ph):
        cur = st.session_state["keys"].get(name, "")
        v = st.text_input(label, value=cur, type="password", placeholder=ph)
        # always store latest (empty clears)
        st.session_state["keys"][name] = v
    provider = st.selectbox("LLM", [
        "OpenAI / gpt-4o-mini","OpenAI / gpt-4o",
        "Anthropic / Claude 3.5","Gemini / 1.5 Pro",
        "Groq / groq-1.0"])
    if provider.startswith("OpenAI"): key("OpenAI API Key","OPENAI_API_KEY","sk-...")
    elif provider.startswith("Anthropic"): key("Anthropic API Key","ANTHROPIC_API_KEY","sk-ant-...")
    elif provider.startswith("Gemini"): key("Google API Key","GOOGLE_API_KEY","AIza...")
    elif provider.startswith("Groq"): key("Groq API Key","GROQ_API_KEY","grq-...")
    if st.button("Reset session"): st.session_state.clear(); st.experimental_rerun()
    mcp_checkbox = st.checkbox("Use MCP tools (Tavily / Neo4j)", value=False)
    privacy_note()

    # Diagnostics: show whether a Tavily key is available (masked) and allow a quick test
    tavily_env_val = env_defaults.get("TAVILY_API_KEY", "")
    tavily_session_val = st.session_state["keys"].get("TAVILY_API_KEY", "")
    def _mask(k: str) -> str:
        return (k[:8] + "...") if k and len(k) > 8 else ("(present)" if k else "(none)")

    if tavily_session_val:
        st.caption(f"Tavily key: loaded in session { _mask(tavily_session_val) }")
    elif tavily_env_val:
        st.caption(f"Tavily key: found in .env { _mask(tavily_env_val) } — will be used")
    else:
        st.caption("Tavily API key is missing. Please set TAVILY_API_KEY in .env or paste it in the sidebar.")

    if st.button("Test Tavily key"):
        # Validate the key (lightweight test using Tavily SDK)
        key_to_test = st.session_state["keys"].get("TAVILY_API_KEY", "") or os.getenv("TAVILY_API_KEY", "")
        if not key_to_test:
            st.error("No Tavily API key provided. Paste it in the sidebar or add it to .env and reload.")
        else:
            try:
                # import locally to avoid unconditional dependency on tavily at import time
                from tavily import TavilyClient, errors as tavily_errors
                client = TavilyClient(api_key=key_to_test)
                # perform a tiny search to validate auth (no heavy work)
                client.search(query="test", max_results=1)
                st.success("Tavily key OK — test search succeeded.")
            except Exception as e:
                # Map known tavily auth error to user-friendly message
                try:
                    if isinstance(e, tavily_errors.InvalidAPIKeyError):
                        st.error("Tavily API key is invalid or unauthorized. Please verify the key and try again.")
                    else:
                        st.error(f"Tavily test failed: {e}")
                except Exception:
                    # If tavily_errors isn't defined or another error occurred, fall back
                    st.error(f"Tavily test failed: {e}")

    # Per-provider test buttons (OpenAI, Anthropic, Groq, Google/Gemini)
    st.markdown("---")
    def _run_llm_test(label: str, key_name: str, display_name: str):
        key_to_test = st.session_state["keys"].get(key_name, "") or os.getenv(key_name, "")
        if not key_to_test:
            st.error(f"No {display_name} key provided. Paste it in the sidebar or add it to .env and reload.")
            return
        try:
            llm = build_llm(label, st.session_state["keys"])
            # Lightweight prompt that expects a short reply
            msg = [{"role": "system", "content": "You are a tiny connectivity test responder."},
                   {"role": "user", "content": "Respond with OK."}]
            resp = llm.chat(msg, stream=False, max_tokens=16)
            # Some adapters return generators or objects; convert to str safely
            out = ''.join(resp) if hasattr(resp, '__iter__') and not isinstance(resp, str) else str(resp)
            if out and ("ok" in out.lower() or "ok" == out.strip().lower() or len(out) > 0):
                st.success(f"{display_name} key OK — test call succeeded.")
            else:
                st.warning(f"{display_name} test returned an unexpected response: {out}")
        except Exception as e:
            st.error(f"{display_name} test failed: {e}")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Test OpenAI key"):
            _run_llm_test("OpenAI / gpt-4o-mini", "OPENAI_API_KEY", "OpenAI")
        if st.button("Test Anthropic key"):
            _run_llm_test("Anthropic / Claude 3.5", "ANTHROPIC_API_KEY", "Anthropic")
    with col_b:
        if st.button("Test Groq key"):
            _run_llm_test("Groq / groq-1.0", "GROQ_API_KEY", "Groq")
        if st.button("Test Google/Gemini key"):
            _run_llm_test("Gemini / 1.5 Pro", "GOOGLE_API_KEY", "Google/Gemini")
cols = st.columns([1, 2, 1])
with cols[1]:
    st.markdown(
        "<div style='display:flex; justify-content:center; align-items:center;'><h1 style='margin:0; font-size:26px;'>OpenScout — RAG with MCP tool layer</h1></div>",
        unsafe_allow_html=True,
    )
    st.write("")


k = 6
use_reranker = True
use_mcp = bool(st.session_state["keys"].get("USE_MCP", False) or mcp_checkbox)
answer_mode = "concise"
temperature = 0.2
max_tokens = 512

# Simple chat history stored in session so the UI resembles ChatGPT
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list[dict(role:str, content:str, name?:str)]

def render_chat_history():
    for m in st.session_state["messages"]:
        if m.get("role") == "user":
            # user message
            try:
                with st.chat_message("user"):
                    st.write(m.get("content", ""))
            except Exception:
                st.markdown(f"**You:** {m.get('content','')}")
        else:
            # assistant message — show name OpenScope bot and content
            try:
                with st.chat_message("assistant"):
                    st.markdown("**OpenScope bot**")
                    st.write(m.get("content", ""))
            except Exception:
                st.markdown("**OpenScope bot**")
                st.write(m.get("content", ""))

render_chat_history()

query = st.chat_input("Ask anything (we'll fetch sources and cite them)...")

if query:
    # Append the user's question to the chat history (avoid duplicates)
    if not st.session_state["messages"] or st.session_state["messages"][-1].get("content") != query or st.session_state["messages"][-1].get("role") != "user":
        st.session_state["messages"].append({"role": "user", "content": query})

    with st.spinner("Searching, fetching, indexing, and answering…"):
        try:
            llm = build_llm(provider, st.session_state["keys"])
            index, conn = get_index_and_db()
            tools = MCPTools(st.session_state["keys"])  # MCP adapters (SDK fallback)

            # Run graph → returns hits (retrieved chunks) and raw pages indexed
            result = app_graph.invoke(State(
                query=query, k=k,
                use_mcp=use_mcp,
                tavily_api_key=st.session_state["keys"].get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY", ""),
                openai_api_key=st.session_state["keys"].get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
                tools=tools
            ))

            if isinstance(result, dict):
                hits = result.get("hits", []) or []
                synthesizer_fn = result.get("synthesizer") or State.synthesizer
            else:
                hits = getattr(result, "hits", []) or []
                synthesizer_fn = getattr(result, "synthesizer", State.synthesizer)

            if use_reranker:
                hits = maybe_rerank(query, hits, top_k=k)

            # Try streaming the assistant reply if the adapter supports it.
            from core.synthesize import _messages
            messages_for_llm = _messages(query, hits, answer_mode)
            answer = ""
            try:
                maybe_stream = llm.chat(messages_for_llm, stream=True, temperature=temperature, max_tokens=max_tokens)
                # If the response is an iterator/generator, stream token-by-token
                if hasattr(maybe_stream, "__iter__") and not isinstance(maybe_stream, str):
                    acc = ""
                    # Render a live assistant message labeled OpenScope bot
                    try:
                        with st.chat_message("assistant"):
                            st.markdown("**OpenScope bot**")
                            placeholder = st.empty()
                            for chunk in maybe_stream:
                                # some adapters yield dicts or objects; coerce to str
                                text = chunk if isinstance(chunk, str) else str(chunk)
                                acc += text
                                placeholder.write(acc)
                    except Exception:
                        # Fallback if chat_message isn't available in this stream context
                        acc = "".join([str(c) for c in maybe_stream])
                    answer = acc
                else:
                    # Not a stream — get full reply
                    resp = maybe_stream
                    answer = ''.join(resp) if hasattr(resp, '__iter__') and not isinstance(resp, str) else str(resp)
            except Exception as e:
                # Streaming failed (adapter may not support stream or error); fall back to non-stream synthesizer
                try:
                    answer, _ = synthesizer_fn(llm, query, hits, mode=answer_mode, temperature=temperature, max_tokens=max_tokens)
                except Exception as e2:
                    raise

            # compute citations similar to synthesize_with_llm
            citations = [{"id": i} for i, _ in enumerate(hits, start=1) if f"[#{i}]" in answer]

            # Append assistant reply to chat history so it appears in the UI
            st.session_state["messages"].append({"role": "assistant", "content": answer})
        except Exception as e:
            # Show a friendly error message with optional dev details in an expander
            st.error("An error occurred while searching or fetching sources.")
            with st.expander("Details (click to expand)"):
                st.write(str(e))
            # Log the exception for debugging
            logger.exception("Error running graph pipeline")
            # Stop further processing
            st.stop()

    # Layout: answer on left (main), compact sources on right
    left, right = st.columns([3, 1])
    with left:
        last_assistant = ""
        if st.session_state.get("messages"):
            last = st.session_state["messages"][-1]
            if last.get("role") == "assistant":
                last_assistant = last.get("content", "") or ""

        if not last_assistant or last_assistant.strip() != (answer or "").strip():
            st.markdown("### Answer")
            st.write(answer)
        else:
            # No duplicate: optionally show a short caption directing users to the chat
            st.markdown("### Answer")
            st.info("Answer shown in the chat above (OpenScope bot).")
    with right:
        st.markdown("### Sources")
        render_sources(hits, citations)
