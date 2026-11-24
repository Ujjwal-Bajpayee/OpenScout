from tavily import TavilyClient, errors as tavily_errors

def tavily_search(query: str, api_key: str, k: int = 8):
    key = api_key.strip().strip("'\"") if api_key else ""
    if not key:
        raise RuntimeError("Tavily API key not provided. Please enter your Tavily API key in the sidebar.")

    client = TavilyClient(api_key=key)
    try:
        res = client.search(query=query, search_depth="advanced", max_results=k)
    except Exception as e:
        if isinstance(e, tavily_errors.InvalidAPIKeyError):
            raise RuntimeError("Tavily API key is invalid or unauthorized. Please verify the key in the sidebar.") from e
        raise RuntimeError(f"Tavily search failed: {e}") from e

    out = []
    for r in res.get("results", []):
        out.append({"url": r.get("url", ""), "title": r.get("title", ""), "snippet": r.get("content", "")})
    return out
