from tavily import TavilyClient, errors as tavily_errors

def tavily_search(query: str, api_key: str, k: int = 8):
    client = TavilyClient(api_key=api_key) if api_key else TavilyClient()
    try:
        res = client.search(query=query, max_results=k)
    except Exception as e:
        if isinstance(e, tavily_errors.InvalidAPIKeyError):
            raise RuntimeError(
                "Tavily API key is missing or invalid. Please set TAVILY_API_KEY in .env or paste it in the sidebar."
            ) from e
        raise RuntimeError(f"Tavily search failed: {e}") from e

    out = []
    for r in res.get("results", []):
        out.append({"url": r.get("url", ""), "title": r.get("title", ""), "snippet": r.get("content", "")})
    return out
