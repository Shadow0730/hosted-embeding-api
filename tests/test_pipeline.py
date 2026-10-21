"""Offline tests for the PDF pipeline and local retrieval behavior."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pymupdf
import numpy as np
from fastapi.testclient import TestClient

from api import create_app
from embedding_providers import EmbedResult
from evaluate import evaluate_provider
from pdf_processor import extract_and_chunk, split_text
from retrieval import index_chunks, retrieve


class FakeProvider:
    model_id = "fake-test-model"

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str], input_type: str = "document") -> EmbedResult:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            if "blue" in text.lower() or "blue" in input_type.lower():
                vectors.append([1.0, 0.0, 0.0])
            elif "green" in text.lower():
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        if input_type == "query":
            vectors = [[1.0, 0.0, 0.0] for _ in texts]
        return EmbedResult(np.asarray(vectors, dtype=np.float32), 3 * len(texts))


class PipelineTests(unittest.TestCase):
    def test_search_api_requires_key_and_returns_source_passages(self):
        chunks = [
            {
                "chunk_id": "one",
                "text": "The agreement runs for one year.",
                "filename": "agreement.pdf",
                "page": 4,
                "chunk_number": 1,
            }
        ]
        provider = FakeProvider()
        config = {
            "settings": {"api_batch_size": 64},
            "models": {"fake": {"price_per_million_tokens": 0.02}},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            index_dir = Path(temporary_directory)
            index_chunks(provider, "fake", chunks, index_dir, 0.02)
            app = create_app(
                config=config,
                index_dir=index_dir,
                providers={"fake": provider},
                api_key="test-secret",
            )
            client = TestClient(app)

            unauthorized = client.post(
                "/v1/search", json={"question": "agreement term", "model": "fake"}
            )
            response = client.post(
                "/v1/search",
                headers={"X-API-Key": "test-secret"},
                json={"question": "agreement term", "model": "fake"},
            )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["matches"][0]["filename"], "agreement.pdf")
        self.assertEqual(payload["matches"][0]["page"], 4)
        self.assertIn("similarity", payload["matches"][0])

    def test_split_text_keeps_overlap(self):
        text = "alpha beta gamma delta epsilon zeta eta theta"
        chunks = split_text(text, chunk_size=20, overlap=5)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunks))
        self.assertIn("beta", chunks[0])

    def test_extracts_page_metadata_and_flags_sparse_page(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            pdf_dir = Path(temporary_directory)
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text(
                (72, 72),
                "This is selectable text on a page and it has enough characters to avoid an OCR warning.",
            )
            document.new_page()
            pdf_path = pdf_dir / "sample.pdf"
            document.save(pdf_path)
            document.close()

            chunks, warnings = extract_and_chunk(
                pdf_dir, chunk_size=200, overlap=20, sparse_page_chars=40
            )
            self.assertEqual(chunks[0]["filename"], "sample.pdf")
            self.assertEqual(chunks[0]["page"], 1)
            self.assertTrue(any("page(s) 2" in warning for warning in warnings))

    def test_index_cache_prevents_repeat_embedding_and_retrieves_top_five(self):
        chunks = [
            {"chunk_id": "one", "text": "blue sky", "filename": "a.pdf", "page": 1},
            {"chunk_id": "two", "text": "green field", "filename": "a.pdf", "page": 2},
        ]
        provider = FakeProvider()
        with tempfile.TemporaryDirectory() as temporary_directory:
            index_dir = Path(temporary_directory)
            first = index_chunks(provider, "fake", chunks, index_dir, 0.02)
            second = index_chunks(provider, "fake", chunks, index_dir, 0.02)
            self.assertEqual(first["newly_embedded_chunks"], 2)
            self.assertEqual(second["newly_embedded_chunks"], 0)
            self.assertEqual(len(provider.calls), 1)

            result = retrieve(provider, "fake", "blue question", index_dir, top_k=5)
            self.assertEqual(result["matches"][0]["filename"], "a.pdf")
            self.assertEqual(result["matches"][0]["page"], 1)
            self.assertGreater(result["matches"][0]["similarity"], 0.99)

    def test_legacy_numpy_cache_migrates_without_reembedding(self):
        chunks = [
            {"chunk_id": "one", "text": "blue sky", "filename": "a.pdf", "page": 1},
            {"chunk_id": "two", "text": "green field", "filename": "a.pdf", "page": 2},
        ]
        provider = FakeProvider()
        with tempfile.TemporaryDirectory() as temporary_directory:
            index_dir = Path(temporary_directory)
            np.savez_compressed(
                index_dir / "fake.npz",
                ids=np.asarray(["one", "two"], dtype=str),
                vectors=np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
            )
            (index_dir / "fake.json").write_text(
                json.dumps(
                    {
                        "model_id": provider.model_id,
                        "chunks": chunks,
                        "billing": {
                            "tokens": 6,
                            "cost_usd_estimate": 0.00000012,
                            "token_sources": ["provider-reported"],
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = index_chunks(provider, "fake", chunks, index_dir, 0.02)
            retrieved = retrieve(provider, "fake", "blue question", index_dir)

            self.assertEqual(result["newly_embedded_chunks"], 0)
            self.assertEqual(result["cached_chunks"], 2)
            self.assertEqual(provider.calls, [["blue question"]])
            self.assertEqual(retrieved["matches"][0]["filename"], "a.pdf")
            self.assertEqual(retrieved["matches"][0]["page"], 1)

    def test_page_labels_produce_hit_rate_at_one_and_five(self):
        chunks = [
            {"chunk_id": "one", "text": "blue sky", "filename": "a.pdf", "page": 1},
            {"chunk_id": "two", "text": "green field", "filename": "a.pdf", "page": 2},
        ]
        provider = FakeProvider()
        with tempfile.TemporaryDirectory() as temporary_directory:
            index_dir = Path(temporary_directory)
            index_chunks(provider, "fake", chunks, index_dir, 0.02)
            metrics = evaluate_provider(
                provider,
                "fake",
                [
                    {
                        "question": "blue question",
                        "supporting_pages": [{"filename": "a.pdf", "page": 2}],
                    }
                ],
                index_dir,
                0.02,
            )

        self.assertEqual(metrics["hit_rate_at_1"], 0.0)
        self.assertEqual(metrics["hit_rate_at_5"], 1.0)


if __name__ == "__main__":
    unittest.main()
