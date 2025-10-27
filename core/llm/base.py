from typing import List, Dict, Any
class LLM:
    name: str
    def chat(self, messages: List[Dict[str,str]], stream: bool=False, **kw) -> Any:
        raise NotImplementedError
