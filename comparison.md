# Embedding model comparison

Prices last checked: 2026-10-15 (see `config.yaml` for official links).

Estimated costs use configured list prices and do not subtract provider credits or free allowances.
Similarity scores are ranking signals, not accuracy percentages. Hit rates below are computed only from your labeled evaluation questions.

| Model | Hit Rate@1 | Hit Rate@5 | Index s | Query embed s (avg) | Local search s (avg) | Index cost (USD, est.) | Query cost (USD, est.) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| openai-small (`text-embedding-3-small`) | 1.0000 | 1.0000 | 0.029 | 0.836 | 0.000215 | 0.000037 | 0.000000 |
| openai-large (`text-embedding-3-large`) | 1.0000 | 1.0000 | 0.086 | 1.105 | 0.000252 | 0.000240 | 0.000001 |
| voyage-4 (`voyage-4`) | 1.0000 | 1.0000 | 0.066 | 1.787 | 0.000216 | 0.000114 | 0.000001 |

Token counts use provider-reported usage when available; character-based fallbacks are marked in `data/metrics.json`. Local search time measures NumPy ranking only.
