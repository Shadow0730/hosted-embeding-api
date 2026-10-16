"""Manual page-label evaluation and CSV/Markdown comparison reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from datetime import datetime, timezone

from retrieval import retrieve


def load_evaluation_questions(path: Path) -> tuple[list[dict], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data.get("questions", [])
    examples = sum(bool(item.get("example_only")) for item in questions)
    labeled = [
        item
        for item in questions
        if not item.get("example_only") and item.get("supporting_pages")
    ]
    return labeled, examples


def evaluate_provider(
    provider,
    model_key: str,
    questions: list[dict],
    index_dir: Path,
    price_per_million_tokens: float,
) -> dict:
    """Calculate Hit Rate@1/@5 using user-supplied filenames and page numbers."""
    if not questions:
        return {
            "labeled_question_count": 0,
            "hit_rate_at_1": None,
            "hit_rate_at_5": None,
            "avg_question_embedding_seconds": None,
            "avg_local_search_seconds": None,
            "query_tokens": 0,
            "query_cost_usd_estimate": 0.0,
            "query_token_sources": [],
        }

    hits_at_1: list[bool] = []
    hits_at_5: list[bool] = []
    embedding_times: list[float] = []
    search_times: list[float] = []
    total_tokens = 0
    total_cost = 0.0
    token_sources: set[str] = set()

    for question in questions:
        result = retrieve(
            provider,
            model_key,
            question["question"],
            index_dir,
            top_k=5,
            price_per_million_tokens=price_per_million_tokens,
        )
        expected = {
            (item["filename"], int(item["page"]))
            for item in question["supporting_pages"]
        }
        retrieved_pages = [
            (match["filename"], int(match["page"])) for match in result["matches"]
        ]
        hits_at_1.append(bool(expected.intersection(retrieved_pages[:1])))
        hits_at_5.append(bool(expected.intersection(retrieved_pages[:5])))
        embedding_times.append(result["question_embedding_seconds"])
        search_times.append(result["local_search_seconds"])
        total_tokens += result["query_tokens"]
        total_cost += result["query_cost_usd_estimate"]
        token_sources.add(result["query_token_source"])

    return {
        "labeled_question_count": len(questions),
        "hit_rate_at_1": mean(hits_at_1),
        "hit_rate_at_5": mean(hits_at_5),
        "avg_question_embedding_seconds": mean(embedding_times),
        "avg_local_search_seconds": mean(search_times),
        "query_tokens": total_tokens,
        "query_cost_usd_estimate": total_cost,
        "query_token_sources": sorted(token_sources),
    }


def save_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def write_comparison_reports(
    report_dir: Path,
    model_settings: dict,
    metrics: dict,
    *,
    price_checked: str,
) -> tuple[Path, Path]:
    """Write simple tabular reports; never fabricate retrieval results."""
    report_dir.mkdir(parents=True, exist_ok=True)
    models = metrics.get("models", {})
    rows: list[dict] = []
    for model_key, model_metrics in models.items():
        settings = model_settings.get(model_key, {})
        rows.append(
            {
                "model": model_key,
                "model_id": settings.get("model_id", model_metrics.get("model_id", "")),
                "labeled_questions": model_metrics.get("labeled_question_count", 0),
                "hit_rate_at_1": model_metrics.get("hit_rate_at_1"),
                "hit_rate_at_5": model_metrics.get("hit_rate_at_5"),
                "indexing_seconds_last_run": model_metrics.get("indexing_seconds"),
                "question_embedding_seconds_avg": model_metrics.get(
                    "avg_question_embedding_seconds"
                ),
                "local_search_seconds_avg": model_metrics.get("avg_local_search_seconds"),
                "index_tokens_cumulative": model_metrics.get("index_tokens_cumulative", 0),
                "query_tokens": model_metrics.get("query_tokens", 0),
                "estimated_index_cost_usd_cumulative": model_metrics.get(
                    "index_cost_usd_estimate_cumulative", 0.0
                ),
                "estimated_query_cost_usd": model_metrics.get(
                    "query_cost_usd_estimate", 0.0
                ),
                "price_per_million_tokens": settings.get("price_per_million_tokens"),
                "price_checked": settings.get("price_checked", price_checked),
            }
        )

    csv_path = report_dir / "comparison.csv"
    fieldnames = list(rows[0]) if rows else [
        "model", "model_id", "labeled_questions", "hit_rate_at_1", "hit_rate_at_5",
        "indexing_seconds_last_run", "question_embedding_seconds_avg",
        "local_search_seconds_avg", "index_tokens_cumulative", "query_tokens",
        "estimated_index_cost_usd_cumulative", "estimated_query_cost_usd",
        "price_per_million_tokens", "price_checked",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = report_dir / "comparison.md"
    table_headers = [
        "Model", "Hit Rate@1", "Hit Rate@5", "Index s", "Query embed s (avg)",
        "Local search s (avg)", "Index cost (USD, est.)", "Query cost (USD, est.)",
    ]
    lines = [
        "# Embedding model comparison",
        "",
        f"Prices last checked: {price_checked} (see `config.yaml` for official links).",
        "",
        "Estimated costs use configured list prices and do not subtract provider credits or free allowances.",
        "Similarity scores are ranking signals, not accuracy percentages. Hit rates below are computed only from your labeled evaluation questions.",
        "",
    ]
    has_labels = any(row["labeled_questions"] for row in rows)
    if not has_labels:
        lines.extend(
            [
                "**Accuracy not measured:** there are no active manual evaluation labels. Add questions with `supporting_pages` in `evaluation_questions.json`, then run `python main.py evaluate`.",
                "Question-embedding and local-search timings are not available until there is at least one labeled question. Indexing timing and cost are still reported.",
                "",
            ]
        )
    lines.append("| " + " | ".join(table_headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(table_headers)) + " |")

    def display(value, places: int = 4) -> str:
        return "N/A" if value is None else f"{value:.{places}f}"

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row['model']} (`{row['model_id']}`)",
                    display(row["hit_rate_at_1"]),
                    display(row["hit_rate_at_5"]),
                    display(row["indexing_seconds_last_run"], 3),
                    display(row["question_embedding_seconds_avg"], 3),
                    display(row["local_search_seconds_avg"], 6),
                    display(row["estimated_index_cost_usd_cumulative"], 6),
                    display(row["estimated_query_cost_usd"], 6),
                ]
            )
            + " |"
        )
    if not rows:
        lines.append("| No configured provider has been indexed yet | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    lines.extend(
        [
            "",
            "Token counts use provider-reported usage when available; character-based fallbacks are marked in `data/metrics.json`. Local search time measures NumPy ranking only.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, markdown_path
