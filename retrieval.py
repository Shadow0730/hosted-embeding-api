"""Per-model vector storage and cosine-similarity search in PostgreSQL (pgvector)."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np
from pgvector.psycopg import register_vector
import psycopg
from psycopg.types.json import Jsonb

from embedding_providers import EmbeddingProvider

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embedding_chunks (
    collection TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    model_key TEXT NOT NULL,
    model_id TEXT NOT NULL,
    document TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR NOT NULL,
    PRIMARY KEY (collection, chunk_id)
)
"""


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


def _collection_name(model_key: str, model_id: str) -> str:
    safe_key = re.sub(r"[^a-zA-Z0-9_-]", "_", model_key).strip("-_") or "model"
    model_hash = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:12]
    return f"model-{safe_key[:40]}-{model_hash}"


def _connect(database_url: str | None) -> psycopg.Connection:
    database_url = database_url or os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "Set DATABASE_URL (for example in .env) to a PostgreSQL database with pgvector."
        )
    connection = psycopg.connect(database_url)
    try:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.execute(_SCHEMA)
        connection.commit()
        register_vector(connection)
    except Exception:
        connection.close()
        raise
    return connection


def _vector_dimension(connection: psycopg.Connection, collection: str) -> int | None:
    row = connection.execute(
        "SELECT vector_dims(embedding) FROM embedding_chunks WHERE collection = %s LIMIT 1",
        (collection,),
    ).fetchone()
    return row[0] if row else None


def _chunk_metadata(chunk: dict) -> dict:
    return {
        key: value
        for key, value in chunk.items()
        if key not in {"chunk_id", "text"}
        and isinstance(value, (str, int, float, bool))
    }


def _upsert(
    connection: psycopg.Connection,
    collection: str,
    model_key: str,
    model_id: str,
    chunks: list[dict],
    vectors: list[np.ndarray],
) -> None:
    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO embedding_chunks
                (collection, chunk_id, model_key, model_id, document, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (collection, chunk_id) DO UPDATE SET
                document = EXCLUDED.document,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding
            """,
            [
                (
                    collection,
                    chunk["chunk_id"],
                    model_key,
                    model_id,
                    chunk["text"],
                    Jsonb(_chunk_metadata(chunk)),
                    np.asarray(vector, dtype=np.float32),
                )
                for chunk, vector in zip(chunks, vectors)
            ],
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
    *,
    database_url: str | None = None,
) -> dict:
    """Persist vectors in PostgreSQL and embed only chunks not already indexed."""
    if not chunks:
        raise ValueError("There are no chunks to index")

    started = time.perf_counter()
    index_dir.mkdir(parents=True, exist_ok=True)
    collection = _collection_name(model_key, provider.model_id)
    with _connect(database_url) as connection:
        old_metadata = _read_metadata(index_dir, model_key, provider.model_id)
        legacy_vectors = _read_legacy_vectors(index_dir, model_key, old_metadata)
        existing_billing = old_metadata.get("billing", {})

        current_ids = {chunk["chunk_id"] for chunk in chunks}
        existing_ids = {
            row[0]
            for row in connection.execute(
                "SELECT chunk_id FROM embedding_chunks WHERE collection = %s",
                (collection,),
            )
        }
        stale_ids = existing_ids - current_ids
        if stale_ids:
            connection.execute(
                "DELETE FROM embedding_chunks WHERE collection = %s AND chunk_id = ANY(%s)",
                (collection, sorted(stale_ids)),
            )
            existing_ids.difference_update(stale_ids)

        legacy_chunks = [
            chunk
            for chunk in chunks
            if chunk["chunk_id"] in legacy_vectors
            and chunk["chunk_id"] not in existing_ids
        ]
        if legacy_chunks:
            _upsert(
                connection,
                collection,
                model_key,
                provider.model_id,
                legacy_chunks,
                [legacy_vectors[chunk["chunk_id"]] for chunk in legacy_chunks],
            )
            existing_ids.update(chunk["chunk_id"] for chunk in legacy_chunks)

        missing = [chunk for chunk in chunks if chunk["chunk_id"] not in existing_ids]
        embedding_started = time.perf_counter()
        newly_embedded_tokens = 0
        token_sources: set[str] = set()
        if missing:
            result = provider.embed([chunk["text"] for chunk in missing], "document")
            embedding_seconds = time.perf_counter() - embedding_started
            if len(result.vectors) != len(missing):
                raise ValueError(
                    f"Provider returned {len(result.vectors)} vectors for {len(missing)} chunks"
                )
            _upsert(
                connection,
                collection,
                model_key,
                provider.model_id,
                missing,
                list(result.vectors),
            )
            newly_embedded_tokens, token_source = _count_tokens(
                [chunk["text"] for chunk in missing], result.tokens
            )
            token_sources.add(token_source)
        else:
            embedding_seconds = time.perf_counter() - embedding_started

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


def retrieve(
    provider: EmbeddingProvider,
    model_key: str,
    query: str,
    index_dir: Path,
    *,
    top_k: int = 5,
    price_per_million_tokens: float = 0.0,
    database_url: str | None = None,
) -> dict:
    """Embed a question and retrieve its nearest passages from PostgreSQL."""
    collection = _collection_name(model_key, provider.model_id)
    with _connect(database_url) as connection:
        expected_dimension = _vector_dimension(connection, collection)
        if expected_dimension is None:
            raise FileNotFoundError(
                f"No index for {model_key}. Run 'python main.py index' first."
            )

        started = time.perf_counter()
        query_result = provider.embed([query], "query")
        question_embedding_seconds = time.perf_counter() - started
        if len(query_result.vectors) != 1:
            raise ValueError("Provider did not return exactly one question vector")
        question_vector = np.asarray(query_result.vectors[0], dtype=np.float32)
        if question_vector.shape[0] != expected_dimension:
            raise ValueError(
                "Question and document vector dimensions do not match; re-index with this model."
            )

        search_started = time.perf_counter()
        # Exact cosine search; <=> is pgvector's cosine distance operator.
        rows = connection.execute(
            """
            SELECT chunk_id, document, metadata, embedding <=> %s AS distance
            FROM embedding_chunks
            WHERE collection = %s
            ORDER BY distance
            LIMIT %s
            """,
            (question_vector, collection, top_k),
        ).fetchall()
        local_search_seconds = time.perf_counter() - search_started

    matches = [
        {
            "chunk_id": chunk_id,
            **(metadata or {}),
            "text": document,
            "similarity": float(1.0 - distance),
        }
        for chunk_id, document, metadata, distance in rows
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
