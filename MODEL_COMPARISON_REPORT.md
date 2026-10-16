# Embedding Model and Project Output Report

This report describes the three configured embedding models, the objects and files produced by the project, and the functions, classes, and libraries used. Prices and model specifications were checked on 2026-10-15.

## Models

| Project key | Provider and model ID | Default vector dimensions | Provider response used by adapter | Configured list price |
|---|---|---:|---|---:|
| `openai-small` | OpenAI `text-embedding-3-small` | 1,536 | `data[].embedding`; token usage from `usage.prompt_tokens` | $0.02 / 1M input tokens |
| `openai-large` | OpenAI `text-embedding-3-large` | 3,072 | `data[].embedding`; token usage from `usage.prompt_tokens` | $0.13 / 1M input tokens |
| `voyage-4` | Voyage AI `voyage-4` | 1,024 (default) | `embeddings`; token usage from `total_tokens` | $0.06 / 1M tokens |
| `voyage-law-2` | Voyage AI `voyage-law-2` | 1,024 | `embeddings`; token usage from `total_tokens` | $0.12 / 1M input tokens |

The two OpenAI models use the same `OPENAI_API_KEY`; Voyage uses `VOYAGE_API_KEY`. Missing keys cause those provider entries to be skipped. Configured prices are list-price estimates and do not subtract free allowances, account credits, discounts, or billing adjustments. Voyage's pricing documentation lists a first-200-million-token free allowance for `voyage-4`; the project's estimate intentionally uses the full list price.

The adapter normalizes each provider's response into an `EmbedResult`: `vectors` is a NumPy `float32` array shaped `(number_of_texts, vector_dimensions)`, and `tokens` is reported token usage when available. Each model has its own cache; vectors from different models are never compared to each other.

## Outputs and Fields

### Retrieved passages

The `retrieve()` function returns a dictionary with these fields:

| Field | Meaning |
|---|---|
| `matches` | Up to five ranked passage records |
| `matches[].chunk_id` | Stable identifier for the source chunk |
| `matches[].text` | Retrieved PDF passage text |
| `matches[].filename` | Source PDF filename |
| `matches[].page` | 1-based source page number |
| `matches[].chunk_number` | Chunk order within the page |
| `matches[].similarity` | NumPy cosine similarity ranking score; not an accuracy percentage |
| `question_embedding_seconds` | Time spent embedding the question through the provider |
| `local_search_seconds` | Time spent in local NumPy ranking |
| `query_tokens` | Provider-reported tokens, or a marked estimate if unavailable |
| `query_token_source` | `provider-reported` or `estimated from characters` |
| `query_cost_usd_estimate` | Estimated question embedding cost at configured list price |

### Saved index files

Each model writes a pair of files to `data/indexes/`:

- `<model>.npz`: `ids` and `vectors` arrays.
- `<model>.json`: `model_id`, a `chunks` array (`chunk_id`, `text`, `filename`, `page`, `chunk_number`), and cumulative `billing` data (`tokens`, `cost_usd_estimate`, `token_sources`).


### Comparison CSV columns

`reports/comparison.csv` contains one row per indexed model and these columns:

| CSV column | Description |
|---|---|
| `model` | Project model key |
| `model_id` | Provider model ID |
| `labeled_questions` | Count of active manual evaluation questions |
| `hit_rate_at_1`, `hit_rate_at_5` | Fraction of labeled questions with a correct supporting page in rank 1 or ranks 1-5; blank when not measured |
| `indexing_seconds_last_run` | Duration of the most recent indexing operation |
| `question_embedding_seconds_avg` | Average provider time to embed labeled questions |
| `local_search_seconds_avg` | Average local vector-ranking time |
| `index_tokens_cumulative` | Cumulative document tokens embedded for the current local cache |
| `query_tokens` | Tokens used for evaluated questions |
| `estimated_index_cost_usd_cumulative` | Cumulative estimated document-embedding cost |
| `estimated_query_cost_usd` | Estimated cost for evaluated questions |
| `price_per_million_tokens` | Configured list price |
| `price_checked` | Date the configured price was checked |

If evaluation labels are absent, hit rates and question timing are unavailable. The Markdown report says accuracy was not measured; it does not create sample results.

## Functions, Classes, and Libraries

| Module | Main functions and classes | Libraries / role |
|---|---|---|
| `pdf_processor.py` | `split_text()`, `extract_and_chunk()` | `pymupdf` extracts selectable PDF page text; Python `hashlib` creates stable chunk IDs |
| `embedding_providers.py` | `EmbedResult`, `EmbeddingProvider`, `OpenAIProvider`, `VoyageProvider`, `create_configured_providers()` | `openai` and `voyageai` call hosted embedding APIs; NumPy stores returned vectors |
| `retrieval.py` | `index_chunks()`, `load_index()`, `retrieve()`; private cache helpers `_cache_paths()`, `_read_cache()`, `_write_cache()`, `_count_tokens()` | NumPy performs cosine similarity; `json`, `pathlib`, and `time` handle local caches and timings |
| `evaluate.py` | `load_evaluation_questions()`, `evaluate_provider()`, `save_metrics()`, `write_comparison_reports()` | Python `csv`, `json`, `statistics`, and `datetime` calculate and export evaluation data |
| `main.py` | `load_project()`, `read_metrics()`, `get_chunks()`, `build_parser()`, `run_index()`, `run_ask()`, `run_evaluate()`, `run_report()`, `main()` | `argparse` supplies CLI commands; `python-dotenv` loads local keys; `PyYAML` reads model configuration |

### Third-party packages

| Package | Use |
|---|---|
| `PyMuPDF` | PDF page text extraction |
| `numpy` | Vector arrays and cosine similarity |
| `openai` | OpenAI embeddings SDK |
| `voyageai` | Voyage AI embeddings SDK |

## Official References

- OpenAI embeddings and default dimensions: [Embeddings guide](https://developers.openai.com/api/docs/guides/embeddings)
- OpenAI list prices: [API pricing](https://developers.openai.com/api/docs/pricing)
- Voyage model names and default dimensions: [Text embeddings guide](https://docs.voyageai.com/docs/embeddings)
- Voyage SDK response and usage: [Python client](https://docs.voyageai.com/docs/api-key-and-installation)
- Voyage list prices and free allowance: [Voyage pricing](https://docs.voyageai.com/docs/pricing)

