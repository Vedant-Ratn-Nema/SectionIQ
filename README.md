# SectionIQ

SectionIQ is a local-first Python library for structured PDF retrieval. It
ingests PDFs into typed evidence blocks, builds hybrid sparse+dense indexes, and
returns grounded answers with page/block citations.

## Features

- Structure-aware ingestion for mixed digital PDFs
- Typed retrieval units: sections, paragraphs, list items, tables, captions,
  warnings, and notes when detectable
- Local BM25 plus local dense vector search out of the box
- Pluggable embedding, vector, reranking, and answer-generation backends
- LLM-backed answers when `OPENAI_API_KEY` is configured
- Evidence-first responses with block and page citations
- Benchmark harnesses for comparing against chunking and hierarchy-first systems

## Quick Start

```python
from sectioniq import SectionIQ

engine = SectionIQ(store_path=".sectioniq")
doc_id = engine.ingest("/path/to/manual.pdf")
engine.build_index()

hits = engine.search("What torque specification is required?", top_k=5)
for hit in hits:
    print(hit.block_type, hit.citation, hit.text_preview)

result = engine.answer("What torque specification is required?", top_k=5)
print(result.answer)
print(engine.get_citations(result))
```

## Configuration

Set `OPENAI_API_KEY` to enable OpenAI embeddings, reranking, and answer
generation. Without an API key, `search()` still works locally, and `answer()`
requires a custom answer generator.

Optional environment variables:

- `SECTIONIQ_EMBEDDING_MODEL`
- `SECTIONIQ_RERANK_MODEL`
- `SECTIONIQ_LLM_MODEL`

## Development

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev,bench]"
python -m pytest
```

## Project Layout

- `src/sectioniq/`: library code
- `tests/`: unit tests
- `examples/`: local examples and manual test runners
- `scripts/`: benchmark and comparison utilities
