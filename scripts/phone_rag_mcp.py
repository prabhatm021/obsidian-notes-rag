"""Stdio MCP server that forwards searches to a remote replica.

Runs on the machine where the Claude client lives; the replica (which holds the
index and does the embedding) is reached over the network. Only the `mcp`
package is required here — everything heavy stays on the replica.

Register as a connector with:
    command: python
    args:    ["/path/to/phone_rag_mcp.py"]
    env:     {"PHONE_RAG_URL": "http://<replica-host>:8100"}
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer
except ModuleNotFoundError:  # mcp 2.x renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer

BASE_URL = os.environ.get("PHONE_RAG_URL", "http://127.0.0.1:8100").rstrip("/")
TIMEOUT = float(os.environ.get("PHONE_RAG_TIMEOUT", "60"))
TOKEN = os.environ.get("PHONE_RAG_TOKEN", "")

mcp = MCPServer("phone-rag")


def _headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    return headers


def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode(),
        headers=_headers(),
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.load(resp)


@mcp.tool()
def search_notes(query: str, limit: int = 10, note_type: Optional[str] = None) -> list[dict]:
    """Search the Obsidian notes replica using semantic similarity.

    Args:
        query: Search query text
        limit: Maximum number of results (default 10)
        note_type: Optional filter - "daily" or "note"

    Returns:
        Matching notes with file path, heading, content excerpt and similarity
    """
    payload: dict = {"query": query, "limit": limit}
    if note_type:
        payload["type"] = note_type
    try:
        return _post("/search", payload)["results"]
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return [{"error": "unauthorized: PHONE_RAG_TOKEN missing or wrong"}]
        return [{"error": f"replica error {e.code}: {e.reason}"}]
    except urllib.error.URLError as e:
        return [{"error": f"replica unreachable at {BASE_URL}: {e}"}]


@mcp.tool()
def replica_status() -> dict:
    """Check whether the notes replica is reachable and how much it holds."""
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=10) as resp:
            return {"url": BASE_URL, **json.load(resp)}
    except urllib.error.URLError as e:
        return {"url": BASE_URL, "ok": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
