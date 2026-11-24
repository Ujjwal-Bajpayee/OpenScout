import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai
from .base import LLM

def fetch_available_gemini_models(api_key: str) -> list[str]:
    clean_key = api_key.strip().strip("'\"") if api_key else ""
    if not clean_key:
        return ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"]
    try:
        genai.configure(api_key=clean_key)
        models = genai.list_models()
        chat_models = []
        for m in models:
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                name = m.name.replace("models/", "")
                if not any(k in name.lower() for k in ["embedding", "imagen", "aqa"]):
                    chat_models.append(name)
        return sorted(chat_models) if chat_models else ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"]
    except Exception:
        return ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"]

class GeminiLLM(LLM):
    name = "Gemini"

    def __init__(self, api_key: str, model="gemini-1.5-pro"):
        clean_key = api_key.strip().strip("'\"") if api_key else ""
        if clean_key:
            genai.configure(api_key=clean_key)
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
