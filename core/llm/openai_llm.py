from openai import OpenAI
from .base import LLM

def fetch_available_openai_models(api_key: str) -> list[str]:
    clean_key = api_key.strip().strip("'\"") if api_key else ""
    if not clean_key:
        return ["gpt-4o-mini", "gpt-4o", "o1-mini", "o3-mini"]
    try:
        client = OpenAI(api_key=clean_key, timeout=10.0)
        resp = client.models.list()
        chat_models = [
            m.id for m in resp.data
            if (m.id.startswith("gpt-") or m.id.startswith("o1") or m.id.startswith("o3") or "chat" in m.id.lower())
            and not any(k in m.id.lower() for k in ["realtime", "audio", "transcription", "tts", "moderation", "instruct", "dall-e", "embedding", "search"])
        ]
        return sorted(chat_models) if chat_models else ["gpt-4o-mini", "gpt-4o", "o1-mini", "o3-mini"]
    except Exception:
        return ["gpt-4o-mini", "gpt-4o", "o1-mini", "o3-mini"]

class OpenAILLM(LLM):
    name = "OpenAI"

    def __init__(self, api_key: str, model="gpt-4o-mini"):
        clean_key = api_key.strip().strip("'\"") if api_key else ""
        self.client = OpenAI(api_key=clean_key) if clean_key else OpenAI()
        self.model = model

    def chat(self, messages, stream=False, **kw):
        resp = self.client.chat.completions.create(model=self.model, messages=messages, stream=stream, **kw)
        if not stream:
            return resp.choices[0].message.content or ""

        def gen():
            for c in resp:
                delta = c.choices[0].delta.content or ""
                if delta:
                    yield delta

        return gen()
