"""
MCPTools exposes a stable tool interface for the app/graph.
This starter implements MCP adapters by wrapping SDKs so everything runs out of the box.
Later, you can replace the internals with a real MCP client without changing call sites.
"""
from tavily import TavilyClient, errors as tavily_errors
import os
from neo4j import GraphDatabase

"""
MCPTools exposes a stable tool interface for the app/graph.

This implementation supports two modes:
 - MCP server mode: if `MCP_URL` is provided (in keys or env), the adapter will call the remote MCP HTTP endpoints
   (`/search`, `/extract`, `/cypher`) and return their JSON responses.
 - Local SDK fallback: if no MCP_URL is configured the adapter wraps local SDKs (Tavily, Neo4j) so the app runs out-of-the-box.

The goal: switch to a real MCP server by setting MCP_URL without changing the rest of the app.
"""

import os
from typing import List
import httpx
from tavily import TavilyClient, errors as tavily_errors
from neo4j import GraphDatabase


class MCPTools:
    def __init__(self, keys: dict):
        self.keys = keys or {}
        # MCP server url (optional). If set, use remote MCP endpoints instead of local SDKs.
        self.mcp_url = self.keys.get("MCP_URL") or os.getenv("MCP_URL")
        self._neo4j_driver = None
        if self.keys.get("NEO4J_URI") and self.keys.get("NEO4J_USERNAME") and self.keys.get("NEO4J_PASSWORD"):
            self._neo4j_driver = GraphDatabase.driver(
                self.keys["NEO4J_URI"], auth=(self.keys["NEO4J_USERNAME"], self.keys["NEO4J_PASSWORD"]) 
            )

    # --- Helper: call remote MCP endpoints if configured ---
    def _call_mcp(self, path: str, payload: dict, timeout: float = 20.0) -> dict:
        if not self.mcp_url:
            raise RuntimeError("MCP_URL not configured")
        url = self.mcp_url.rstrip("/") + path
        headers = {"Content-Type": "application/json"}
        # Allow passing an API key to the MCP server if present (optional)
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

    # --- MCP search ---
    def search(self, query: str, k: int = 8) -> List[dict]:
        # If MCP server configured, call it first
        if self.mcp_url:
            payload = {"query": query, "k": k}
            res = self._call_mcp("/search", payload)
            # Expecting {'results': [{'url':..., 'title':..., 'content':...}, ...]}
            out = []
            for r in res.get("results", []):
                out.append({"url": r.get("url", ""), "title": r.get("title", ""), "snippet": r.get("content", "")})
            return out

        # Local SDK fallback
        client = TavilyClient(api_key=self.keys.get("TAVILY_API_KEY")) if self.keys.get("TAVILY_API_KEY") else TavilyClient()
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

    def extract(self, urls: List[str]) -> List[dict]:
        # remote MCP: call /extract which should return cleaned page objects
        if self.mcp_url:
            payload = {"urls": urls}
            res = self._call_mcp("/extract", payload)
            # Expecting {'pages': [{'url':..., 'title':..., 'text':...}, ...]}
            return res.get("pages", [])

        # Local fallback: return URL shells — the fetcher will download and clean them.
        return [{"url": u} for u in urls]

    def cypher(self, query: str, params: dict | None = None) -> List[dict]:
        # remote MCP: call /cypher to execute graph queries centrally
        if self.mcp_url:
            payload = {"query": query, "params": params or {}}
            res = self._call_mcp("/cypher", payload)
            return res.get("rows", [])

        # Local Neo4j fallback
        if not self._neo4j_driver:
            return []
        with self._neo4j_driver.session() as session:
            res = session.run(query, **(params or {}))
            return [r.data() for r in res]
