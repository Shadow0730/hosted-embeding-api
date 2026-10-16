"""Small adapters around the hosted embedding SDKs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Protocol

import numpy as np


@dataclass
class EmbedResult:
    vectors: np.ndarray
    tokens: int | None


class EmbeddingProvider(Protocol):
    model_id: str

    def embed(self, texts: list[str], input_type: str) -> EmbedResult:
        """Return vectors in input order and provider-reported token usage."""


class OpenAIProvider:
    def __init__(self, model_id: str, api_key: str, batch_size: int = 64):
        from openai import OpenAI

        self.model_id = model_id
        self.batch_size = batch_size
        self.client = OpenAI(api_key=api_key)

    def embed(self, texts: list[str], input_type: str = "document") -> EmbedResult:
        vectors: list[list[float]] = []
        total_tokens = 0
        usage_available = True

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model_id,
                input=batch,
                encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
            if response.usage is None:
                usage_available = False
            else:
                total_tokens += response.usage.prompt_tokens

        return EmbedResult(
            vectors=np.asarray(vectors, dtype=np.float32),
            tokens=total_tokens if usage_available else None,
        )


class VoyageProvider:
    def __init__(self, model_id: str, api_key: str, batch_size: int = 64):
        import voyageai

        self.model_id = model_id
        self.batch_size = batch_size
        self.client = voyageai.Client(api_key=api_key, max_retries=2, timeout=60)

    def embed(self, texts: list[str], input_type: str = "document") -> EmbedResult:
        vectors: list[list[float]] = []
        total_tokens = 0
        usage_available = True

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embed(
                texts=batch,
                model=self.model_id,
                input_type=input_type,
            )
            vectors.extend(response.embeddings)
            if response.total_tokens is None:
                usage_available = False
            else:
                total_tokens += response.total_tokens

        return EmbedResult(
            vectors=np.asarray(vectors, dtype=np.float32),
            tokens=total_tokens if usage_available else None,
        )


def create_configured_providers(
    model_settings: dict, batch_size: int = 64
) -> tuple[dict[str, EmbeddingProvider], list[str]]:
    """Create only providers whose API keys are present in the environment."""
    providers: dict[str, EmbeddingProvider] = {}
    skipped: list[str] = []

    for key, settings in model_settings.items():
        api_key = os.getenv(settings["api_key_env"], "").strip()
        if not api_key:
            skipped.append(key)
            continue

        common = (settings["model_id"], api_key, batch_size)
        if settings["provider"] == "openai":
            providers[key] = OpenAIProvider(*common)
        elif settings["provider"] == "voyage":
            providers[key] = VoyageProvider(*common)
        else:
            raise ValueError(f"Unknown provider '{settings['provider']}' for {key}")

    return providers, skipped
