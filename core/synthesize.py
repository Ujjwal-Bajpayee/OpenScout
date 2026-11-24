from typing import List, Dict, Tuple
from .synth_prompt import SYSTEM_PROMPT

def _ctx(hits: List[Dict]) -> str:
    parts = []
    for i, h in enumerate(hits, start=1):
        parts.append(f"[#{i}] {h.get('title', '')}\nURL: {h.get('url', '')}\n{(h.get('text', '')[:1500])}")
    return "\n\n".join(parts)

def _messages(question: str, hits: List[Dict], mode: str = "concise", word_budget: int = 250, history: List[Dict] = None):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for m in history[-4:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                msgs.append({"role": m["role"], "content": m["content"]})
    user_prompt = f"question: {question}\nanswer_mode: {mode}\nword_budget: {word_budget}\ntop_passages:\n{_ctx(hits)}"
    msgs.append({"role": "user", "content": user_prompt})
    return msgs

def synthesize_with_llm(llm, question: str, hits: List[Dict], mode: str = "concise",
                        temperature: float = 0.2, max_tokens: int = 512, history: List[Dict] = None) -> Tuple[str, List[Dict]]:
    out = llm.chat(_messages(question, hits, mode, history=history), stream=False, temperature=temperature, max_tokens=max_tokens)
    used = [{"id": i} for i, _ in enumerate(hits, start=1) if f"[#{i}]" in out]
    return out, used
