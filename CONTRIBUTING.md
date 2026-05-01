# Contributing to SectionIQ

Thanks for helping make structured PDF retrieval less brittle.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev,bench]"
python3 -m pytest
```

## Development Guidelines

- Keep retrieval evidence-first: answers should be grounded in returned blocks.
- Do not make hierarchy a hard gate for retrieval; use it as context and a ranking signal.
- Keep local stores, PDFs, spreadsheets, notebooks, and benchmark outputs out of commits.
- Add tests for retrieval behavior when changing ingestion, indexing, ranking, or answer generation.
- Prefer provider-neutral interfaces for embeddings, vector search, reranking, and answer generation.

## Pull Requests

Please include:

- A short description of the retrieval behavior changed.
- Any new dependencies or provider assumptions.
- Tests or benchmark notes showing why the change helps.
