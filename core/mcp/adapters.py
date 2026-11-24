from typing import List
import httpx
from tavily import TavilyClient, errors as tavily_errors
from neo4j import GraphDatabase

class MCPTools:
    def __init__(self, keys: dict):
        self.keys = keys or {}
        self.mcp_url = self.keys.get("MCP_URL")
        self._neo4j_driver = None
        if self.keys.get("NEO4J_URI") and self.keys.get("NEO4J_USERNAME") and self.keys.get("NEO4J_PASSWORD"):
            try:
                self._neo4j_driver = GraphDatabase.driver(
                    self.keys["NEO4J_URI"], auth=(self.keys["NEO4J_USERNAME"], self.keys["NEO4J_PASSWORD"])
                )
            except Exception:
                self._neo4j_driver = None

    def _call_mcp(self, path: str, payload: dict, timeout: float = 20.0) -> dict:
        if not self.mcp_url:
            raise RuntimeError("MCP_URL not configured in sidebar.")
        url = self.mcp_url.rstrip("/") + path
        headers = {"Content-Type": "application/json"}
        if self.keys.get("MCP_API_KEY"):
            headers["Authorization"] = f"Bearer {self.keys.get('MCP_API_KEY')}"
        try:
            r = httpx.post(url, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"MCP server returned HTTP {e.response.status_code}: {e.response.text}") from e
        except Exception as e:
            raise RuntimeError(f"MCP call failed: {e}") from e

    def search(self, query: str, k: int = 8) -> List[dict]:
        if self.mcp_url:
            payload = {"query": query, "k": k}
            res = self._call_mcp("/search", payload)
            out = []
            for r in res.get("results", []):
                out.append({"url": r.get("url", ""), "title": r.get("title", ""), "snippet": r.get("content", "")})
            return out

        tavily_key = (self.keys.get("TAVILY_API_KEY") or "").strip().strip("'\"")
        if not tavily_key:
            raise RuntimeError("Tavily API key not provided. Please enter your Tavily API key in the sidebar.")

        client = TavilyClient(api_key=tavily_key)
        try:
            res = client.search(query=query, max_results=k)
        except Exception as e:
            if isinstance(e, tavily_errors.InvalidAPIKeyError):
                raise RuntimeError("Tavily API key is invalid or unauthorized. Please verify the key in the sidebar.") from e
            raise RuntimeError(f"Tavily search failed: {e}") from e

        out = []
        for r in res.get("results", []):
            out.append({"url": r.get("url", ""), "title": r.get("title", ""), "snippet": r.get("content", "")})
        return out

    def extract(self, urls: List[str]) -> List[dict]:
        if self.mcp_url:
            payload = {"urls": urls}
            res = self._call_mcp("/extract", payload)
            return res.get("pages", [])
        return [{"url": u} for u in urls]

    def cypher(self, query: str, params: dict | None = None) -> List[dict]:
        if self.mcp_url:
            payload = {"query": query, "params": params or {}}
            res = self._call_mcp("/cypher", payload)
            return res.get("rows", [])

        if not self._neo4j_driver:
            return []
        with self._neo4j_driver.session() as session:
            res = session.run(query, **(params or {}))
            return [r.data() for r in res]
