"""HTTP API for retrieving source passages from indexed legal documents."""

from __future__ import annotations

import hmac
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
import yaml

from embedding_providers import create_configured_providers
from retrieval import retrieve

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    model: str
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedPassage(BaseModel):
    chunk_id: str
    filename: str
    page: int
    chunk_number: int | None = None
    text: str
    similarity: float


class SearchResponse(BaseModel):
    query: str
    model: str
    model_id: str
    matches: list[RetrievedPassage]
    question_embedding_seconds: float
    local_search_seconds: float
    query_tokens: int
    query_token_source: str
    query_cost_usd_estimate: float


def create_app(
    *,
    config: dict | None = None,
    index_dir: Path | None = None,
    providers: dict | None = None,
    api_key: str | None = None,
) -> FastAPI:
    """Build the API, allowing tests to inject local config and fake providers."""
    load_dotenv(PROJECT_ROOT / ".env")
    if config is None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    settings = config["settings"]
    if index_dir is None:
        index_dir = PROJECT_ROOT / settings["index_dir"]
    if api_key is None:
        api_key = os.getenv("LEGALAI_API_KEY", "").strip()
    if providers is None:
        providers, _ = create_configured_providers(
            config["models"], settings["api_batch_size"]
        )

    app = FastAPI(title="LegalAI Retrieval API", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/search", response_model=SearchResponse)
    def search(
        request: SearchRequest,
        x_api_key: str | None = Header(default=None),
    ) -> dict:
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="Service authentication is not configured",
            )
        if not x_api_key or not hmac.compare_digest(x_api_key, api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        if request.model not in config["models"]:
            raise HTTPException(status_code=422, detail="Unknown model")

        provider = providers.get(request.model)
        if provider is None:
            raise HTTPException(
                status_code=503,
                detail="The requested embedding provider is not configured",
            )

        model_settings = config["models"][request.model]
        try:
            result = retrieve(
                provider,
                request.model,
                request.question.strip(),
                index_dir,
                top_k=request.top_k,
                price_per_million_tokens=model_settings["price_per_million_tokens"],
            )
        except FileNotFoundError as error:
            raise HTTPException(
                status_code=503,
                detail="Search index is not ready; index documents first",
            ) from error
        except Exception as error:
            raise HTTPException(
                status_code=502,
                detail="The retrieval provider could not complete the search",
            ) from error

        return {
            "query": request.question.strip(),
            "model": request.model,
            "model_id": provider.model_id,
            **result,
        }

    return app


app = create_app()