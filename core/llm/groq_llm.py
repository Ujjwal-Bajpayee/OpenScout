import httpx
from .base import LLM

class GroqLLM(LLM):
    name = "Groq"
    def __init__(self, api_key: str, model: str = "groq-1.0"):
        self.api_key = api_key
        self.model = model

    def chat(self, messages, stream=False, **kw):
        # Build a simple prompt from messages
        prompt_parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt_parts.append(f"{role.upper()}: {content}")
        prompt = "\n".join(prompt_parts)

        if not self.api_key:
            # Graceful fallback when no key is provided
            return (
                "[Groq adapter] No GROQ_API_KEY provided. Provide the key in the sidebar to use Groq,\n"
                "or select a different LLM.\n\n" +
                "Request prompt:\n" + prompt[:1000]
            )

        url = kw.get("endpoint") or "https://api.groq.ai/v1/generate"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {"model": self.model, "input": prompt}

        try:
            r = httpx.post(url, json=data, headers=headers, timeout=30.0)
            r.raise_for_status()
            j = r.json()
            # Try common response fields
            if isinstance(j, dict):
                return j.get("text") or j.get("output") or j.get("result") or str(j)
            return str(j)
        except Exception as e:
            return f"[Groq adapter] request failed: {e}"
