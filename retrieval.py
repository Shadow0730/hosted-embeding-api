"""Per-model vector caching and NumPy cosine-similarity search."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import time

import chromadb
import numpy as np

from embedding_providers import EmbeddingProvider


def _cache_paths(index_dir: Path, model_key: str) -> tuple[Path, Path]:
    safe_key = re.sub(r"[^a-zA-Z0-9_-]", "_", model_key)
    return index_dir / f"{safe_key}.npz", index_dir / f"{safe_key}.json"


def _read_metadata(index_dir: Path, model_key: str, model_id: str) -> dict:
    vector_path, metadata_path = _cache_paths(index_dir, model_key)
    if not metadata_path.exists():
        return {}

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model_id") != model_id:
            return {}
        return metadata
    except (OSError, ValueError):
        return {}


def _read_legacy_vectors(index_dir: Path, model_key: str, metadata: dict) -> dict:
    vector_path, _ = _cache_paths(index_dir, model_key)
    if not vector_path.exists() or not metadata:
        return {}

    try:
        with np.load(vector_path, allow_pickle=False) as stored:
            ids = stored["ids"].astype(str).tolist()
            vectors = stored["vectors"].astype(np.float32)
        if len(ids) != len(vectors):
            return {}
        return {chunk_id: vector for chunk_id, vector in zip(ids, vectors)}
    except (OSError, ValueError, KeyError):
        return {}


def _write_metadata(
    index_dir: Path,
    model_key: str,
    model_id: str,
    chunks: list[dict],
    billing: dict,
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    _, metadata_path = _cache_paths(index_dir, model_key)
    metadata_path.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "chunks": chunks,
                "billing": billing,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _open_collection(index_dir: Path, model_key: str, model_id: str):
    safe_key = re.sub(r"[^a-zA-Z0-9_-]", "_", model_key).strip("-_") or "model"
    model_hash = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:12]
    collection_name = f"model-{safe_key[:40]}-{model_hash}"
    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "model_key": model_key,
            "model_id": model_id,
            "hnsw:space": "cosine",
        },
    )
    return client, collection


def _chunk_metadata(chunk: dict) -> dict:
    return {
        key: value
        for key, value in chunk.items()
        if key not in {"chunk_id", "text"}
        and isinstance(value, (str, int, float, bool))
    }


def _count_tokens(texts: list[str], reported_tokens: int | None) -> tuple[int, str]:
    if reported_tokens is not None:
        return reported_tokens, "provider-reported"
    # A rough fallback for providers or SDK versions that omit usage metadata.
    return sum(max(1, math.ceil(len(text) / 4)) for text in texts), "estimated from characters"


def index_chunks(
    provider: EmbeddingProvider,
    model_key: str,
    chunks: list[dict],
    index_dir: Path,
    price_per_million_tokens: float,
) -> dict:
    """Persist vectors in Chroma and embed only chunks not already indexed."""
    if not chunks:
        raise ValueError("There are no chunks to index")

    started = time.perf_counter()
    index_dir.mkdir(parents=True, exist_ok=True)
    client, collection = _open_collection(index_dir, model_key, provider.model_id)
    try:
        old_metadata = _read_metadata(index_dir, model_key, provider.model_id)
        legacy_vectors = _read_legacy_vectors(index_dir, model_key, old_metadata)
        existing_billing = old_metadata.get("billing", {})

        current_ids = {chunk["chunk_id"] for chunk in chunks}
        existing_ids = set(collection.get(include=[])["ids"])
        stale_ids = existing_ids - current_ids
        if stale_ids:
            collection.delete(ids=sorted(stale_ids))
            existing_ids.difference_update(stale_ids)

        legacy_chunks = [
            chunk
            for chunk in chunks
            if chunk["chunk_id"] in legacy_vectors
            and chunk["chunk_id"] not in existing_ids
        ]
        if legacy_chunks:
            collection.upsert(
                ids=[chunk["chunk_id"] for chunk in legacy_chunks],
                embeddings=[
                    legacy_vectors[chunk["chunk_id"]].tolist()
                    for chunk in legacy_chunks
                ],
                documents=[chunk["text"] for chunk in legacy_chunks],
                metadatas=[_chunk_metadata(chunk) for chunk in legacy_chunks],
            )
            existing_ids.update(chunk["chunk_id"] for chunk in legacy_chunks)

        missing = [chunk for chunk in chunks if chunk["chunk_id"] not in existing_ids]
        embedding_started = time.perf_counter()
        newly_embedded_tokens = 0
        token_sources: set[str] = set()
        vector_dimension = collection.metadata.get("vector_dimension")
        if missing:
            result = provider.embed([chunk["text"] for chunk in missing], "document")
            embedding_seconds = time.perf_counter() - embedding_started
            if len(result.vectors) != len(missing):
                raise ValueError(
                    f"Provider returned {len(result.vectors)} vectors for {len(missing)} chunks"
                )
            vector_dimension = int(result.vectors.shape[1])
            collection.upsert(
                ids=[chunk["chunk_id"] for chunk in missing],
                embeddings=result.vectors.astype(np.float32).tolist(),
                documents=[chunk["text"] for chunk in missing],
                metadatas=[_chunk_metadata(chunk) for chunk in missing],
            )
            newly_embedded_tokens, token_source = _count_tokens(
                [chunk["text"] for chunk in missing], result.tokens
            )
            token_sources.add(token_source)
        else:
            embedding_seconds = time.perf_counter() - embedding_started

        if vector_dimension is not None:
            collection_metadata = {
                key: value
                for key, value in (collection.metadata or {}).items()
                if key != "hnsw:space"
            }
            collection.modify(
                metadata={
                    **collection_metadata,
                    "vector_dimension": vector_dimension,
                }
            )

        cumulative_tokens = int(existing_billing.get("tokens", 0)) + newly_embedded_tokens
        cumulative_cost = float(existing_billing.get("cost_usd_estimate", 0.0)) + (
            newly_embedded_tokens / 1_000_000 * price_per_million_tokens
        )
        cumulative_sources = set(existing_billing.get("token_sources", [])) | token_sources
        billing = {
            "tokens": cumulative_tokens,
            "cost_usd_estimate": cumulative_cost,
            "token_sources": sorted(cumulative_sources),
        }
        _write_metadata(index_dir, model_key, provider.model_id, chunks, billing)

        return {
            "model_id": provider.model_id,
            "chunk_count": len(chunks),
            "newly_embedded_chunks": len(missing),
            "cached_chunks": len(chunks) - len(missing),
            "indexing_seconds": time.perf_counter() - started,
            "embedding_seconds": embedding_seconds,
            "index_tokens_cumulative": cumulative_tokens,
            "index_cost_usd_estimate_cumulative": cumulative_cost,
            "token_sources": sorted(cumulative_sources),
            "price_per_million_tokens": price_per_million_tokens,
        }
    finally:
        client.close()


def retrieve(
    provider: EmbeddingProvider,
    model_key: str,
    query: str,
    index_dir: Path,
    *,
    top_k: int = 5,
    price_per_million_tokens: float = 0.0,
) -> dict:
    """Embed a question and retrieve its nearest passages from local Chroma."""
    client, collection = _open_collection(index_dir, model_key, provider.model_id)
    try:
        chunk_count = collection.count()
        if not chunk_count:
            raise FileNotFoundError(
                f"No index for {model_key}. Run 'python main.py index' first."
            )

        started = time.perf_counter()
        query_result = provider.embed([query], "query")
        question_embedding_seconds = time.perf_counter() - started
        if len(query_result.vectors) != 1:
            raise ValueError("Provider did not return exactly one question vector")
        question_vector = query_result.vectors[0]
        expected_dimension = collection.metadata.get("vector_dimension")
        if expected_dimension is not None and question_vector.shape[0] != expected_dimension:
            raise ValueError(
                "Question and document vector dimensions do not match; re-index with this model."
            )

        search_started = time.perf_counter()
        results = collection.query(
            query_embeddings=[question_vector.astype(np.float32).tolist()],
            n_results=min(top_k, chunk_count),
            include=["documents", "metadatas", "distances"],
        )
        local_search_seconds = time.perf_counter() - search_started
    finally:
        client.close()

    matches = [
        {
            "chunk_id": chunk_id,
            **(metadata or {}),
            "text": document,
            "similarity": float(1.0 - distance),
        }
        for chunk_id, document, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
    token_count, token_source = _count_tokens([query], query_result.tokens)
    return {
        "matches": matches,
        "question_embedding_seconds": question_embedding_seconds,
        "local_search_seconds": local_search_seconds,
        "query_tokens": token_count,
        "query_token_source": token_source,
        "query_cost_usd_estimate": token_count / 1_000_000 * price_per_million_tokens,
    }
