from openai import OpenAI
from .base import LLM

class OpenAILLM(LLM):
    name = "OpenAI"
    def __init__(self, api_key: str, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
    def chat(self, messages, stream=False, **kw):
        resp = self.client.chat.completions.create(model=self.model, messages=messages, stream=stream, **kw)
        if not stream: return resp.choices[0].message.content or ""
        def gen():
            for c in resp:
                delta = c.choices[0].delta.content or ""
                if delta: yield delta
        return gen()
