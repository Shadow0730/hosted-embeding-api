# Compare PDF Embedding Models

A small, step-by-step Python project for learning a retrieval pipeline:

```text
PDF files -> page text -> shared text chunks -> hosted embeddings -> PostgreSQL (pgvector)
                                                           question -> top passages
```

It compares two OpenAI embedding models and one Voyage AI model. It does not generate answers or run a web app. PostgreSQL with the pgvector extension stores and searches the vectors; retrieval results are the PDF passages themselves.

For the guided, cell-by-cell learning path, open [`embedding_model_comparison.ipynb`](embedding_model_comparison.ipynb) in Jupyter or VS Code. It uses the same modules and settings as the CLI below. Run cells in order; the indexing, question, and evaluation cells can incur hosted API charges.

For a summary of model dimensions, API outputs, project functions/classes, libraries, and CSV fields, see [`MODEL_COMPARISON_REPORT.md`](MODEL_COMPARISON_REPORT.md).

## 1. Set Up Python

Python 3.10 or newer is recommended. From the project folder, create and activate a virtual environment.

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**macOS or Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`PyMuPDF` extracts selectable PDF text, `psycopg` and `pgvector` store vectors in PostgreSQL and run cosine search, and the two provider SDKs call embedding APIs.

## 2. Add PDFs and API Keys

Copy `.env.example` to `.env`, then fill in only the services you want to compare:

```dotenv
OPENAI_API_KEY=your_openai_key
VOYAGE_API_KEY=your_voyage_key
DATABASE_URL=postgresql://user:password@localhost:5432/legalai
```

`DATABASE_URL` must point to a PostgreSQL database where the [pgvector](https://github.com/pgvector/pgvector) extension is installed. The project runs `CREATE EXTENSION IF NOT EXISTS vector` and creates its `embedding_chunks` table on first use, so the database user needs permission to do that (or an administrator can create the extension once). The quickest local setup is the official pgvector Docker image:

```bash
docker run -d --name legalai-db -e POSTGRES_USER=legalai -e POSTGRES_PASSWORD=legalai -e POSTGRES_DB=legalai -p 5432:5432 pgvector/pgvector:pg17
```

which matches `DATABASE_URL=postgresql://legalai:legalai@localhost:5432/legalai`.

Get API keys from the [OpenAI API platform](https://platform.openai.com/api-keys) and [Voyage AI dashboard](https://dashboard.voyageai.com/organization/api-keys). An unset key skips that provider. Never commit `.env` or share it; it is excluded by `.gitignore`.

Place up to 10 `.pdf` files in `data/pdfs/`. The project sorts files by name and processes the first 10. Your PDFs and generated indexes are ignored by Git because they may contain private content.

## 3. Index the PDFs

```bash
python main.py index
```

For each provider with a configured key, the program:

1. Extracts text page by page and remembers the filename and 1-based page number.
2. Warns when a page has little or no selectable text. Scanned pages need OCR before this project can read them.
3. Splits extracted pages into shared overlapping chunks. Every model sees the same chunk text.
4. Sends only uncached chunks to each configured embedding provider.
5. Stores each model's vectors in the PostgreSQL `embedding_chunks` table, kept separate per model.

Existing `.npz` indexes in `data/indexes/` are imported into PostgreSQL on the next `index` run without re-embedding their chunks. Per-model billing totals stay in the `.json` files there.

The cache key is based on the chunk text, filename, page, chunk order, and model ID. Re-running index does not repeat embedding calls for unchanged chunks. Changed chunks are sent again. Changing chunk size or overlap changes the chunks and can require new paid requests.

## 4. Ask a Question

First index the PDFs, then choose a configured model key:

```bash
python main.py ask --model openai-small --question "What does the document say about this topic?"
```

Other model keys are `openai-large` and `voyage-4`. The program embeds the question with the selected model, asks PostgreSQL for that model's nearest vectors, and prints up to five passages with filenames, pages, and similarity scores.

Similarity scores rank results; they are **not accuracy percentages**. There is no answer-generation step.

## 5. Serve Search Results to Other Services

The HTTP API returns ranked PDF passages with filenames and page numbers. It does not generate a legal answer. Configure a service-to-service key in `.env`:

```dotenv
LEGALAI_API_KEY=use-a-long-random-secret
```

Start the API from the project folder after indexing the PDFs:

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Other services can call `POST /v1/search` with the key in the `X-API-Key` header:

```powershell
$headers = @{ "X-API-Key" = $env:LEGALAI_API_KEY }
$body = @{
  question = "What is the agreement term?"
  model = "openai-small"
  top_k = 5
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://localhost:8000/v1/search" `
  -Headers $headers -ContentType "application/json" -Body $body
```

The JSON response includes `matches` containing passage text, filename, page, chunk number, and similarity, plus query timing and estimated usage. `top_k` must be between 1 and 20. Use HTTPS and store the key in a secret manager in deployed environments; do not commit it or send it to browser clients. `GET /health` is an unauthenticated liveness check. API documentation is available at `/docs` while the service is running.

## 6. Add Evaluation Labels

Edit `evaluation_questions.json`. Replace the clearly marked example with questions answerable from your own PDFs. For every question, list at least one filename and 1-based page that contains a supporting passage:

```json
{
  "questions": [
    {
      "question": "What does the guide say about safe storage?",
      "supporting_pages": [
        {"filename": "guide.pdf", "page": 3}
      ]
    }
  ]
}
```

A correct supporting page is counted as a hit if it appears at rank 1 (Hit Rate@1) or in the first five results (Hit Rate@5). Use real questions and check their supporting pages yourself. The example entry has `example_only: true` and is ignored, so this project does not invent evaluation results.

## 7. Evaluate and Export Reports

```bash
python main.py evaluate
```

This indexes any new chunks, evaluates each configured provider using the labeled questions, then writes:

- `reports/comparison.csv`
- `reports/comparison.md`

To regenerate the reports from saved metrics without calling a provider:

```bash
python main.py report
```

If no active labels are present, the report shows `N/A` for hit rates and states that accuracy cannot yet be measured. It still reports indexing cost and timing. Question-embedding and local-search timings require at least one labeled question.

## What the Measurements Mean

- **Indexing time** includes the measured indexing operation; it is close to zero when chunks are fully cached.
- **Question-embedding time** is the time spent on the provider call for an evaluation question.
- **Local search time** measures the PostgreSQL vector query, not embedding-provider network time.
- **Token counts** use provider-reported usage when available. Otherwise the project marks a rough character-based estimate in `data/metrics.json`.
- **Estimated API cost** is based on configured per-million-token list prices and observed or estimated token counts. Estimates do not subtract account credits, free-token allowances, discounts, taxes, or other billing adjustments. Provider invoices are authoritative.
- Index cost in the report is cumulative in the local metrics file, not a billing statement.

Prices in `config.yaml` were checked on **2026-10-15** against these official sources:

- [OpenAI API pricing](https://developers.openai.com/api/docs/pricing): `text-embedding-3-small` at $0.02 / 1M input tokens; `text-embedding-3-large` at $0.13 / 1M input tokens.
- [Voyage AI pricing](https://docs.voyageai.com/docs/pricing): `voyage-4` at $0.06 / 1M tokens, with a published first-200M-token account allowance. The estimate intentionally shows list-price cost before any allowance.
- Model IDs and SDK request formats were checked against the [OpenAI embeddings guide](https://developers.openai.com/api/docs/guides/embeddings), [Voyage embedding guide](https://docs.voyageai.com/docs/embeddings), and [Voyage Python client guide](https://docs.voyageai.com/docs/api-key-and-installation).

## Project Files

- `main.py`: CLI commands for indexing, questions, evaluation, and report generation.
- `pdf_processor.py`: PyMuPDF page extraction and shared overlapping chunks.
- `embedding_providers.py`: small OpenAI and Voyage adapters.
- `retrieval.py`: PostgreSQL/pgvector storage, legacy-cache migration, and retrieval timings.
- `evaluate.py`: page-hit evaluation and CSV/Markdown export.
- `config.yaml`: model IDs, chunk settings, and documented prices.
- `evaluation_questions.json`: editable manual labels.
- `data/pdfs/`: input PDFs.
- `data/indexes/`: per-model chunk and billing metadata (and any legacy `.npz` caches); excluded from Git because it may contain private document text. Chunk text and vectors live in PostgreSQL.
- `tests/test_pipeline.py`: tests using a generated PDF and a fake provider; no paid API calls.

## Run the Offline Tests

```bash
python -m unittest discover -s tests -v
```

The retrieval and API tests need a disposable PostgreSQL database with pgvector; they skip unless `TEST_DATABASE_URL` is set. They drop and recreate the `embedding_chunks` table, so never point it at your real database:

```bash
TEST_DATABASE_URL=postgresql://legalai:legalai@localhost:5432/legalai_test python -m unittest discover -s tests -v
```
