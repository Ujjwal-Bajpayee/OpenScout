from groq import Groq
from .base import LLM

def fetch_available_groq_models(api_key: str) -> list[str]:
    clean_key = api_key.strip().strip("'\"") if api_key else ""
    if not clean_key:
        return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    try:
        client = Groq(api_key=clean_key, timeout=10.0)
        resp = client.models.list()
        chat_models = [
            m.id for m in resp.data
            if not any(k in m.id.lower() for k in ["whisper", "guard", "distil", "vision-preview", "tool-use"])
        ]
        return sorted(chat_models) if chat_models else ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    except Exception:
        return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

class GroqLLM(LLM):
    name = "Groq"

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key.strip().strip("'\"") if api_key else ""
        self.model = model
        self.client = Groq(api_key=self.api_key, timeout=30.0) if self.api_key else None

    def chat(self, messages, stream: bool = False, **kw):
        if not self.api_key or not self.client:
            raise RuntimeError("Groq API key not provided. Please enter your Groq API key in the sidebar.")

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                **kw
            )
        except Exception as e:
            err_msg = str(e).lower()
            if "decommissioned" in err_msg or "not_found" in err_msg or "does not exist" in err_msg:
                raise RuntimeError(
                    f"The Groq model '{self.model}' is decommissioned or not found. Please choose an active model in the sidebar."
                ) from e
            raise

        if not stream:
            return resp.choices[0].message.content or ""

        def gen():
            for c in resp:
                delta = c.choices[0].delta.content or ""
                if delta:
                    yield delta

        return gen()
