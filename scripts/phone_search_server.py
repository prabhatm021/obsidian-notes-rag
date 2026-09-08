"""Read-only search server for a replica of the index (numpy only, no sqlite-vec).

Runs on a machine that only holds a synced copy of the portable DB
(`obsidian-rag export-portable`) plus a local llama-server for embedding the
query. Serves plain JSON so it needs nothing heavier than numpy — the MCP layer
lives on whichever machine runs the Claude client.

    python phone_search_server.py --db ~/obsidian-rag-data/portable.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import struct
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

EMBED_PREFIX_QUERY = "search_query: "  # nomic-embed expects task prefixes


class Index:
    def __init__(self, db_path: str, embed_url: str, model: str):
        self.embed_url = embed_url
        self.model = model

        db = sqlite3.connect(db_path)
        rows = db.execute(
            "SELECT id, file_path, heading, type, content, embedding FROM chunks"
            " WHERE embedding IS NOT NULL"
        ).fetchall()
        db.close()
        if not rows:
            raise SystemExit(f"no embedded chunks in {db_path}")

        self.meta = [r[:5] for r in rows]
        dim = len(rows[0][5]) // 4
        mat = np.empty((len(rows), dim), dtype=np.float32)
        for i, r in enumerate(rows):
            mat[i] = struct.unpack(f"{dim}f", r[5])

        # Normalize once so cosine similarity is a single matrix-vector product.
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.mat = mat / norms

    def embed_query(self, text: str) -> np.ndarray:
        payload = json.dumps(
            {"model": self.model, "input": f"{EMBED_PREFIX_QUERY}{text}"}
        ).encode()
        req = urllib.request.Request(
            f"{self.embed_url}/v1/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
        vec = np.asarray(data["data"][0]["embedding"], dtype=np.float32)
        n = float(np.linalg.norm(vec))
        return vec / (n or 1.0)

    def search(self, query: str, limit: int = 10, note_type: str | None = None) -> list[dict]:
        sims = self.mat @ self.embed_query(query)

        # Over-fetch before filtering so a type filter can still fill `limit`.
        take = min(len(sims), limit * 10 if note_type else limit)
        top = np.argpartition(-sims, take - 1)[:take]
        top = top[np.argsort(-sims[top])]

        out = []
        for i in top:
            _id, file_path, heading, ctype, content = self.meta[i]
            if note_type and ctype != note_type:
                continue
            out.append({
                "file_path": file_path,
                "heading": heading or None,
                "content": content[:500],
                "similarity": round(float(sims[i]), 3),
                "type": ctype or "note",
            })
            if len(out) >= limit:
                break
        return out


def make_handler(index: Index):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict):
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "chunks": len(index.meta)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/search":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
                results = index.search(
                    req["query"],
                    limit=int(req.get("limit", 10)),
                    note_type=req.get("type"),
                )
            except Exception as e:
                self._send(400, {"error": str(e)})
                return
            self._send(200, {"results": results})

        def log_message(self, *args):
            pass  # quiet: this runs as a background service

    return Handler


def _selftest():
    idx = Index.__new__(Index)
    idx.meta = [("a", "a.md", "", "note", "alpha"), ("b", "b.md", "", "daily", "beta")]
    idx.mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    idx.embed_query = lambda _q: np.array([1.0, 0.0], dtype=np.float32)

    hits = idx.search("x", limit=1)
    assert hits[0]["file_path"] == "a.md", hits
    assert hits[0]["similarity"] == 1.0, hits

    typed = idx.search("x", limit=1, note_type="daily")
    assert typed[0]["file_path"] == "b.md", typed
    print("selftest ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=False, help="path to portable DB")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--embed-url", default="http://127.0.0.1:8090")
    p.add_argument("--model", default="nomic-embed-text-v1.5")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.db:
        p.error("--db is required")

    index = Index(args.db, args.embed_url.rstrip("/"), args.model)
    print(f"loaded {len(index.meta)} chunks, serving on {args.host}:{args.port}")
    HTTPServer((args.host, args.port), make_handler(index)).serve_forever()


if __name__ == "__main__":
    main()
