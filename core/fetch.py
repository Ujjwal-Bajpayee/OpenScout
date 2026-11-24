import httpx
import trafilatura
import asyncio
import urllib.parse

async def _fetch_one(client: httpx.AsyncClient, item: str | dict) -> dict:
    url = item if isinstance(item, str) else item.get("url", "")
    fallback_title = item.get("title", "") if isinstance(item, dict) else ""
    fallback_text = item.get("snippet", "") if isinstance(item, dict) else ""

    try:
        r = await client.get(url, timeout=15)
        r.raise_for_status()
        html = r.text
        meta = trafilatura.extract_metadata(html)
        title = meta.title if (meta and getattr(meta, "title", None)) else (fallback_title or url)
        text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
        if not text.strip() and fallback_text:
            text = fallback_text
        domain = urllib.parse.urlparse(url).netloc
        return {"url": url, "title": title, "text": text, "domain": domain}
    except Exception:
        domain = urllib.parse.urlparse(url).netloc if url else ""
        return {"url": url, "title": fallback_title or url, "text": fallback_text, "domain": domain}

async def fetch_many(items: list) -> list:
    if not items:
        return []
    async with httpx.AsyncClient(follow_redirects=True, headers={"User-Agent": "OpenScout/1.0"}) as client:
        return await asyncio.gather(*[_fetch_one(client, u) for u in items])
