import anthropic
from .base import LLM

def fetch_available_anthropic_models(api_key: str) -> list[str]:
    clean_key = api_key.strip().strip("'\"") if api_key else ""
    default_models = ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus-latest"]
    if not clean_key:
        return default_models
    try:
        client = anthropic.Anthropic(api_key=clean_key, timeout=10.0)
        resp = client.models.list()
        models = [m.id for m in resp.data]
        return sorted(models) if models else default_models
    except Exception:
        return default_models

class AnthropicLLM(LLM):
    name = "Anthropic"

    def __init__(self, api_key: str, model="claude-3-5-sonnet-latest"):
        clean_key = api_key.strip().strip("'\"") if api_key else ""
        self.client = anthropic.Anthropic(api_key=clean_key) if clean_key else anthropic.Anthropic()
        self.model = model

    def chat(self, messages, stream=False, **kw):
        sys = ""
        conv = []
        for m in messages:
            if m["role"] == "system":
                sys = m["content"]
            else:
                conv.append({"role": m["role"], "content": m["content"]})

        if stream:
            s = self.client.messages.stream(model=self.model, system=sys, messages=conv, **kw)
            def gen():
                with s as stream_resp:
                    for ev in stream_resp.text_stream:
                        yield ev
            return gen()

        r = self.client.messages.create(model=self.model, system=sys, messages=conv, **kw)
        return "".join([b.text for b in r.content if getattr(b, "type", "") == "text"])
