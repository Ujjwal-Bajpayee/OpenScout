import google.generativeai as genai
from .base import LLM

class GeminiLLM(LLM):
    name = "Gemini"

    def __init__(self, api_key: str, model="gemini-1.5-pro"):
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def chat(self, messages, stream=False, **kw):
        sys = "\n".join([m["content"] for m in messages if m["role"] == "system"])
        text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages if m["role"] != "system"])

        if stream:
            s = self.model.generate_content([sys, text], stream=True, **kw)
            def gen():
                for c in s:
                    if c.text:
                        yield c.text
            return gen()

        r = self.model.generate_content([sys, text], **kw)
        return r.text or ""
