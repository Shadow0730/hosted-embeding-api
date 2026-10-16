"""Command-line entry point for the PDF embedding comparison tutorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from dotenv import load_dotenv
import yaml

from embedding_providers import create_configured_providers
from evaluate import (
    evaluate_provider,
    load_evaluation_questions,
    save_metrics,
    write_comparison_reports,
)
from pdf_processor import extract_and_chunk
from retrieval import index_chunks, retrieve

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_project() -> tuple[dict, dict[str, Path]]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    settings = config["settings"]
    paths = {
        name: PROJECT_ROOT / settings[name]
        for name in (
            "pdf_dir", "index_dir", "report_dir", "evaluation_file", "metrics_file"
        )
    }
    load_dotenv(PROJECT_ROOT / ".env")
    return config, paths


def read_metrics(path: Path) -> dict:
    if not path.exists():
        return {"models": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def get_chunks(config: dict, paths: dict[str, Path]) -> list[dict]:
    settings = config["settings"]
    chunks, warnings = extract_and_chunk(
        paths["pdf_dir"],
        max_pdfs=settings["max_pdfs"],
        chunk_size=settings["chunk_size"],
        overlap=settings["chunk_overlap"],
        sparse_page_chars=settings["sparse_page_chars"],
    )
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Created {len(chunks)} shared chunks from {paths['pdf_dir']}.")
    return chunks


def build_parser(model_keys: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed local PDF chunks, retrieve passages, and compare models."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index", help="Extract PDFs and index each configured provider")

    ask_parser = commands.add_parser("ask", help="Retrieve passages for one question")
    ask_parser.add_argument("--model", required=True, choices=model_keys)
    ask_parser.add_argument("--question", required=True)

    commands.add_parser("evaluate", help="Evaluate all configured providers and write reports")
    commands.add_parser("report", help="Regenerate CSV and Markdown reports from saved metrics")
    return parser


def run_index(config: dict, paths: dict[str, Path]) -> int:
    settings = config["settings"]
    providers, skipped = create_configured_providers(
        config["models"], settings["api_batch_size"]
    )
    for model_key in skipped:
        print(f"Skipping {model_key}: {config['models'][model_key]['api_key_env']} is not set.")

    chunks = get_chunks(config, paths)
    if not providers:
        print("No provider API keys are configured. Add a key to .env and run index again.")
        return 0

    metrics = read_metrics(paths["metrics_file"])
    metrics.setdefault("models", {})
    for model_key, provider in providers.items():
        model_config = config["models"][model_key]
        print(f"Indexing {model_key} ({provider.model_id})...")
        index_metrics = index_chunks(
            provider,
            model_key,
            chunks,
            paths["index_dir"],
            model_config["price_per_million_tokens"],
        )
        previous = metrics["models"].get(model_key, {})
        previous.update(index_metrics)
        metrics["models"][model_key] = previous
        print(
            f"  {index_metrics['newly_embedded_chunks']} new, "
            f"{index_metrics['cached_chunks']} cached; "
            f"{index_metrics['indexing_seconds']:.2f}s total, "
            f"estimated cumulative index cost ${index_metrics['index_cost_usd_estimate_cumulative']:.6f}."
        )

    metrics["skipped_providers"] = skipped
    save_metrics(paths["metrics_file"], metrics)
    return 0


def run_ask(args: argparse.Namespace, config: dict, paths: dict[str, Path]) -> int:
    model_key = args.model
    model_config = config["models"][model_key]
    providers, skipped = create_configured_providers(
        {model_key: model_config}, config["settings"]["api_batch_size"]
    )
    if model_key not in providers:
        print(
            f"Cannot query {model_key}: set {model_config['api_key_env']} in .env first.",
            file=sys.stderr,
        )
        return 2

    result = retrieve(
        providers[model_key],
        model_key,
        args.question,
        paths["index_dir"],
        top_k=5,
        price_per_million_tokens=model_config["price_per_million_tokens"],
    )
    print(f"\nTop passages for {model_key} ({model_config['model_id']}):")
    for rank, match in enumerate(result["matches"], start=1):
        print(
            f"\n[{rank}] {match['filename']} page {match['page']} "
            f"| cosine similarity {match['similarity']:.4f}"
        )
        print(match["text"])
    print("\nSimilarity is a ranking score, not an accuracy percentage.")
    print(
        f"Question embedding: {result['question_embedding_seconds']:.3f}s; "
        f"local search: {result['local_search_seconds']:.6f}s; "
        f"{result['query_tokens']} tokens ({result['query_token_source']}); "
        f"estimated API cost ${result['query_cost_usd_estimate']:.8f}."
    )
    return 0


def run_evaluate(config: dict, paths: dict[str, Path]) -> int:
    settings = config["settings"]
    providers, skipped = create_configured_providers(
        config["models"], settings["api_batch_size"]
    )
    for model_key in skipped:
        print(f"Skipping {model_key}: {config['models'][model_key]['api_key_env']} is not set.")
    if not providers:
        print("No providers are configured. Add API keys to .env and try again.")
        return 0

    chunks = get_chunks(config, paths)
    questions, example_count = load_evaluation_questions(paths["evaluation_file"])
    if example_count:
        print(f"Ignoring {example_count} example-only evaluation question(s).")
    if not questions:
        print(
            "No labeled questions found. Models will be indexed, but retrieval hit rates "
            "cannot be measured until supporting_pages are filled in."
        )

    metrics = {"models": {}, "skipped_providers": skipped}
    for model_key, provider in providers.items():
        model_config = config["models"][model_key]
        print(f"\nPreparing {model_key} ({provider.model_id})...")
        index_metrics = index_chunks(
            provider,
            model_key,
            chunks,
            paths["index_dir"],
            model_config["price_per_million_tokens"],
        )
        retrieval_metrics = evaluate_provider(
            provider,
            model_key,
            questions,
            paths["index_dir"],
            model_config["price_per_million_tokens"],
        )
        metrics["models"][model_key] = {**index_metrics, **retrieval_metrics}
        print(
            f"Indexing {index_metrics['indexing_seconds']:.2f}s; "
            f"new chunks {index_metrics['newly_embedded_chunks']}; "
            f"estimated cumulative index cost "
            f"${index_metrics['index_cost_usd_estimate_cumulative']:.6f}."
        )
        if questions:
            print(
                f"Hit Rate@1: {retrieval_metrics['hit_rate_at_1']:.1%}; "
                f"Hit Rate@5: {retrieval_metrics['hit_rate_at_5']:.1%}."
            )

    save_metrics(paths["metrics_file"], metrics)
    price_checked = max(
        model["price_checked"] for model in config["models"].values()
    )
    csv_path, markdown_path = write_comparison_reports(
        paths["report_dir"], config["models"], metrics, price_checked=price_checked
    )
    print(f"\nSaved {csv_path} and {markdown_path}.")
    return 0


def run_report(config: dict, paths: dict[str, Path]) -> int:
    metrics = read_metrics(paths["metrics_file"])
    price_checked = max(
        model["price_checked"] for model in config["models"].values()
    )
    csv_path, markdown_path = write_comparison_reports(
        paths["report_dir"], config["models"], metrics, price_checked=price_checked
    )
    print(f"Saved {csv_path} and {markdown_path}.")
    return 0


def main() -> int:
    config, paths = load_project()
    parser = build_parser(list(config["models"]))
    args = parser.parse_args()
    if args.command == "index":
        return run_index(config, paths)
    if args.command == "ask":
        return run_ask(args, config, paths)
    if args.command == "evaluate":
        return run_evaluate(config, paths)
    return run_report(config, paths)


if __name__ == "__main__":
    raise SystemExit(main())
