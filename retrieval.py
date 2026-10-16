"""Per-model vector caching and NumPy cosine-similarity search."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time

import numpy as np

from embedding_providers import EmbeddingProvider


def _cache_paths(index_dir: Path, model_key: str) -> tuple[Path, Path]:
    safe_key = re.sub(r"[^a-zA-Z0-9_-]", "_", model_key)
    return index_dir / f"{safe_key}.npz", index_dir / f"{safe_key}.json"


def _read_cache(index_dir: Path, model_key: str, model_id: str) -> tuple[dict, dict]:
    vector_path, metadata_path = _cache_paths(index_dir, model_key)
    if not vector_path.exists() or not metadata_path.exists():
        return {}, {}

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model_id") != model_id:
            return {}, {}
        with np.load(vector_path, allow_pickle=False) as stored:
            ids = stored["ids"].astype(str).tolist()
            vectors = stored["vectors"].astype(np.float32)
        if len(ids) != len(vectors):
            return {}, {}
        cached = {chunk_id: vector for chunk_id, vector in zip(ids, vectors)}
        return cached, metadata
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return {}, {}


def _write_cache(
    index_dir: Path,
    model_key: str,
    model_id: str,
    chunks: list[dict],
    vectors: np.ndarray,
    billing: dict,
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    vector_path, metadata_path = _cache_paths(index_dir, model_key)
    np.savez_compressed(
        vector_path,
        ids=np.asarray([chunk["chunk_id"] for chunk in chunks], dtype=str),
        vectors=np.asarray(vectors, dtype=np.float32),
    )
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
    """Reuse cached chunk vectors and embed only chunks not already indexed."""
    if not chunks:
        raise ValueError("There are no chunks to index")

    started = time.perf_counter()
    cached, old_metadata = _read_cache(index_dir, model_key, provider.model_id)
    existing_billing = old_metadata.get("billing", {})
    vectors_by_id = {
        chunk_id: vector for chunk_id, vector in cached.items()
    }
    missing = [chunk for chunk in chunks if chunk["chunk_id"] not in vectors_by_id]

    embedding_started = time.perf_counter()
    newly_embedded_tokens = 0
    token_sources: set[str] = set()
    if missing:
        result = provider.embed([chunk["text"] for chunk in missing], "document")
        if len(result.vectors) != len(missing):
            raise ValueError(
                f"Provider returned {len(result.vectors)} vectors for {len(missing)} chunks"
            )
        for chunk, vector in zip(missing, result.vectors):
            vectors_by_id[chunk["chunk_id"]] = vector
        newly_embedded_tokens, token_source = _count_tokens(
            [chunk["text"] for chunk in missing], result.tokens
        )
        token_sources.add(token_source)
    embedding_seconds = time.perf_counter() - embedding_started

    ordered_vectors = np.asarray(
        [vectors_by_id[chunk["chunk_id"]] for chunk in chunks], dtype=np.float32
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
    _write_cache(index_dir, model_key, provider.model_id, chunks, ordered_vectors, billing)

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


def load_index(index_dir: Path, model_key: str, model_id: str) -> tuple[list[dict], np.ndarray]:
    vector_path, metadata_path = _cache_paths(index_dir, model_key)
    if not vector_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"No index for {model_key}. Run 'python main.py index' first."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_id") != model_id:
        raise ValueError(
            f"The cached model ID for {model_key} differs from config.yaml; re-index it."
        )
    with np.load(vector_path, allow_pickle=False) as stored:
        vectors = stored["vectors"].astype(np.float32)
    chunks = metadata["chunks"]
    if len(chunks) != len(vectors):
        raise ValueError(f"The saved index for {model_key} is incomplete; re-index it.")
    return chunks, vectors


def retrieve(
    provider: EmbeddingProvider,
    model_key: str,
    query: str,
    index_dir: Path,
    *,
    top_k: int = 5,
    price_per_million_tokens: float = 0.0,
) -> dict:
    """Embed one question, then rank only this model's local vectors."""
    chunks, vectors = load_index(index_dir, model_key, provider.model_id)
    started = time.perf_counter()
    query_result = provider.embed([query], "query")
    question_embedding_seconds = time.perf_counter() - started
    if len(query_result.vectors) != 1:
        raise ValueError("Provider did not return exactly one question vector")

    search_started = time.perf_counter()
    question_vector = query_result.vectors[0]
    if vectors.shape[1] != question_vector.shape[0]:
        raise ValueError(
            "Question and document vector dimensions do not match; re-index with this model."
        )
    denominator = np.linalg.norm(vectors, axis=1) * np.linalg.norm(question_vector)
    dot_products = vectors @ question_vector
    scores = np.divide(
        dot_products,
        denominator,
        out=np.zeros_like(dot_products, dtype=np.float32),
        where=denominator != 0,
    )
    best_indices = np.argsort(scores)[::-1][: min(top_k, len(chunks))]
    local_search_seconds = time.perf_counter() - search_started

    matches = [
        {
            **chunks[int(index)],
            "similarity": float(scores[int(index)]),
        }
        for index in best_indices
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
