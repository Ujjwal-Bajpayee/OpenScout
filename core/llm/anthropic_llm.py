import anthropic
from .base import LLM
class AnthropicLLM(LLM):
    name = "Anthropic"
    def __init__(self, api_key: str, model="claude-3-5-sonnet-latest"):
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
    def chat(self, messages, stream=False, **kw):
        sys = ""; conv=[]
        for m in messages:
            if m["role"]=="system": sys=m["content"]
            else: conv.append({"role": m["role"], "content": m["content"]})
        if stream:
            s = self.client.messages.stream(model=self.model, system=sys, messages=conv, **kw)
            def gen():
                with s as stream_resp:
                    for ev in stream_resp.text_stream: yield ev
            return gen()
        r = self.client.messages.create(model=self.model, system=sys, messages=conv, **kw)
        return "".join([b.text for b in r.content if getattr(b,"type","")== "text"])
