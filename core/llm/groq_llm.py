from groq import Groq
from .base import LLM

class GroqLLM(LLM):
    name = "Groq"

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key.strip().strip("'\"") if api_key else ""
        self.model = model
        self.client = Groq(api_key=self.api_key, timeout=30.0) if self.api_key else None

    def chat(self, messages, stream: bool = False, **kw):
        if not self.api_key or not self.client:
            raise RuntimeError("Groq API key not provided. Please enter your Groq API key in the sidebar.")

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            **kw
        )

        if not stream:
            return resp.choices[0].message.content or ""

        def gen():
            for c in resp:
                delta = c.choices[0].delta.content or ""
                if delta:
                    yield delta

        return gen()
