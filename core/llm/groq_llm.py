from openai import OpenAI
from .base import LLM

class GroqLLM(LLM):
    name = "Groq"

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key if api_key else "dummy_key",
        )

    def chat(self, messages, stream: bool = False, **kw):
        if not self.api_key:
            raise RuntimeError(
                "Groq API key not provided. Set GROQ_API_KEY in .env or paste it in the sidebar."
            )

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
