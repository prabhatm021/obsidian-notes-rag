"""MCP server for obsidian-rag with semantic search tools."""

from __future__ import annotations

import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .config import load_config, Config
from .indexer import create_embedder, Embedder, VaultIndexer, is_lmstudio_running, is_ollama_running
from .store import VectorStore

# Create MCP server
mcp = FastMCP("obsidian-rag")

# Global instances (lazy initialized)
_config: Optional[Config] = None
_embedder: Optional[Embedder] = None
_store: Optional[VectorStore] = None


def get_config() -> Config:
    """Get or create config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_embedder() -> Embedder:
    """Get or create embedder instance."""
    global _embedder
    if _embedder is None:
        config = get_config()
        # Set API key in environment if configured
        if config.provider == "openai" and config.openai_api_key:
            os.environ["OPENAI_API_KEY"] = config.openai_api_key
        
        # Determine model, base_url and api_key based on provider
        if config.provider == "openai":
            model = config.openai_model
            base_url = None
            api_key = config.get_openai_api_key()
        elif config.provider == "ollama":
            model = config.ollama_model
            base_url = config.ollama_url
            api_key = config.get_ollama_api_key()
        else:  # lmstudio
            model = config.lmstudio_model
            base_url = config.lmstudio_url
            api_key = config.get_lmstudio_api_key()

        _embedder = create_embedder(
            provider=config.provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
    return _embedder


def get_store() -> VectorStore:
    """Get or create store instance."""
    global _store
    if _store is None:
        config = get_config()
        _store = VectorStore(data_path=config.get_data_path())
    return _store


@mcp.tool()
def search_notes(
    query: str,
    limit: Optional[int] = None,
    note_type: Optional[str] = None
) -> list[dict]:
    """Search notes using semantic similarity.

    Args:
        query: Search query text
        limit: Maximum number of results (default: from config)
        note_type: Optional filter - "daily" or "note"

    Returns:
        List of matching notes with content, file path, and similarity score
    """
    config = get_config()
    embedder = get_embedder()
    store = get_store()

    # Use config default if caller did not provide a limit
    if limit is None:
        limit = config.indexer.default_search_limit

    # Generate query embedding
    query_embedding = embedder.embed(query, task_type="search_query")

    # Build filter
    where = {"type": note_type} if note_type else None

    # Search
    results = store.search(query_embedding, limit=limit, where=where)

    # Apply similarity threshold from config
    threshold = config.indexer.similarity_threshold

    # Format results
    return [
        {
            "file_path": r["metadata"]["file_path"],
            "heading": r["metadata"].get("heading") or None,
            "content": r["content"][:500] if len(r["content"]) > 500 else r["content"],
            "similarity": round(1 - r["distance"], 3),
            "type": r["metadata"].get("type", "note")
        }
        for r in results
        if threshold <= 0 or (1 - r["distance"]) >= threshold
    ]


@mcp.tool()
def get_similar(note_path: str, limit: Optional[int] = None) -> list[dict]:
    """Find notes similar to the given note.

    Args:
        note_path: Path to the note (relative to vault root)
        limit: Number of similar notes to return (default: from config)

    Returns:
        List of similar notes with content preview and similarity score
    """
    config = get_config()
    embedder = get_embedder()
    store = get_store()

    # Use config default if caller did not provide a limit
    if limit is None:
        limit = config.indexer.default_similar_limit

    # Get all chunks from this note by direct lookup
    results = store.get_by_file(note_path)

    if not results:
        return [{"error": f"Note not found: {note_path}"}]

    # Combine content from all chunks of this note
    note_content = "\n\n".join(r["content"] for r in results)

    # Generate embedding for the note content
    note_embedding = embedder.embed(note_content[:8000])  # Limit for embedding

    # Search for similar notes, excluding the source note
    all_results = store.search(note_embedding, limit=limit + 10)

    # Filter out chunks from the same file
    similar = [
        r for r in all_results
        if r["metadata"]["file_path"] != note_path
    ][:limit]

    return [
        {
            "file_path": r["metadata"]["file_path"],
            "heading": r["metadata"].get("heading") or None,
            "preview": r["content"][:200] if len(r["content"]) > 200 else r["content"],
            "similarity": round(1 - r["distance"], 3)
        }
        for r in similar
    ]


@mcp.tool()
def get_note_context(note_path: str, limit: Optional[int] = None) -> dict:
    """Get a note and its related context.

    Args:
        note_path: Path to the note (relative to vault root)
        limit: Number of similar notes to include (default: from config)

    Returns:
        Note content and list of similar notes for context
    """
    config = get_config()
    store = get_store()

    # Use config default if caller did not provide a limit
    if limit is None:
        limit = config.indexer.default_context_limit

    # Get all chunks from this file by direct lookup
    results = store.get_by_file(note_path)

    if not results:
        return {"error": f"Note not found: {note_path}"}

    # Combine chunks to get full note content
    note_content = "\n\n".join(r["content"] for r in results)

    # Get similar notes
    similar = get_similar(note_path, limit=limit)

    return {
        "file_path": note_path,
        "content": note_content,
        "similar_notes": similar if not (similar and "error" in similar[0]) else []
    }


@mcp.tool()
def get_stats() -> dict:
    """Get index statistics.

    Returns:
        Statistics about the indexed notes collection
    """
    store = get_store()
    return store.get_stats()


@mcp.tool()
def health_check() -> dict:
    """Check whether the index is in sync with the vault and the embedding backend is reachable.

    Read-only: does not modify the index. Compares every file in the vault against
    what the store has recorded as indexed (via the same mtime tracking `reindex`
    uses), so it catches both files the watcher/reindex missed and stale entries
    left behind by deleted files.

    Returns:
        Provider connectivity, chunk/file counts, and any stale or ghost files found
    """
    config = get_config()
    store = get_store()

    if not config.vault_path:
        return {"error": "No vault path configured. Run 'obsidian-rag setup' first."}

    if config.provider == "ollama":
        embedder_reachable = is_ollama_running(config.ollama_url, config.get_ollama_api_key())
    elif config.provider == "lmstudio":
        embedder_reachable = is_lmstudio_running(config.lmstudio_url, config.get_lmstudio_api_key())
    else:
        embedder_reachable = bool(config.get_openai_api_key())

    indexer = VaultIndexer(vault_path=config.vault_path, config=config.indexer)
    drift = indexer.check_drift(store)

    return {
        "provider": config.provider,
        "embedder_reachable": embedder_reachable,
        "total_chunks": store.get_stats()["count"],
        "vault_files": drift["vault_files"],
        "tracked_files": drift["tracked_files"],
        "in_sync": drift["in_sync"],
        "stale_files": len(drift["stale_files"]),
        "stale_file_examples": drift["stale_files"][:10],
        "ghost_files": len(drift["ghost_files"]),
        "ghost_file_examples": drift["ghost_files"][:10],
    }


@mcp.tool()
def reindex(clear: bool = False, path_filter: Optional[str] = None) -> dict:
    """Re-index the Obsidian vault.

    Args:
        clear: If True, clear existing index before re-indexing (default: False)
        path_filter: Optional path prefix to limit indexing (e.g., "Daily Notes/")

    Returns:
        Statistics about the indexing operation
    """
    config = get_config()
    embedder = get_embedder()
    store = get_store()

    if not config.vault_path:
        return {"error": "No vault path configured. Run 'obsidian-rag setup' first."}

    indexer = VaultIndexer(vault_path=config.vault_path, embedder=embedder, config=config.indexer)

    result = indexer.index_vault(store, clear=clear, path_filter=path_filter)

    return {
        "files_indexed": result["files_indexed"],
        "files_skipped": result["files_skipped"],
        "files_removed": result["files_removed"],
        "chunks_created": result["chunks_created"],
        "total_in_store": store.get_stats()["count"],
        "errors": result["errors"] if result["errors"] else None,
        "path_filter": path_filter,
        "cleared": clear
    }


def run_server(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000):
    """Run the MCP server.

    transport "stdio" (default) is for a local Claude Desktop/Code connector.
    "streamable-http" exposes the server over the network (e.g. via Tailscale)
    so a remote Claude client can add it as a connector.
    """
    if transport != "stdio":
        mcp.settings.host = host
        mcp.settings.port = port
    mcp.run(transport=transport)  # type: ignore[arg-type]


if __name__ == "__main__":
    run_server()
