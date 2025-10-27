import httpx, trafilatura, asyncio, urllib.parse

async def _fetch_one(client, url):
    try:
        r = await client.get(url, timeout=15)
        r.raise_for_status()
        html = r.text
        title = ""
        meta = trafilatura.extract_metadata(html)
        if meta and getattr(meta, "title", None): title = meta.title
        text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
        domain = urllib.parse.urlparse(url).netloc
        return {"url": url, "title": title or url, "text": text, "domain": domain}
    except Exception:
        return {"url": url, "title": url, "text": "", "domain": ""}

async def fetch_many(urls):
    async with httpx.AsyncClient(follow_redirects=True, headers={"User-Agent":"OpenScout/1.0"}) as client:
        return await asyncio.gather(*[_fetch_one(client,u) for u in urls])
