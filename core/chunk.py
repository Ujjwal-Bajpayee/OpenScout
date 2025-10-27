from .constants import CHUNK_TOKENS, CHUNK_OVERLAP
import re

def _sents(text:str): return re.split(r'(?<=[.!?])\s+', text.strip())

def chunk_text(text: str, target_tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP):
    sents = _sents(text)
    chunks, cur, count = [], [], 0
    for s in sents:
        tok = max(1, len(s.split()))
        if count + tok > target_tokens and cur:
            chunks.append(" ".join(cur))
            # overlap tail
            back, bt = [], 0
            for t in reversed(cur):
                tt = max(1, len(t.split()))
                if bt + tt > overlap: break
                bt += tt; back.insert(0, t)
            cur, count = back[:], sum(len(x.split()) for x in back)
        cur.append(s); count += tok
    if cur: chunks.append(" ".join(cur))
    return chunks
