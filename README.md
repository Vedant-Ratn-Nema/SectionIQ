# SectionIQ

SectionIQ is a local-first Python library for structured PDF retrieval. It
ingests PDFs into typed evidence blocks, builds hybrid sparse+dense indexes, and
returns grounded answers with page/block citations.

The core design is deliberately not tree-first: hierarchy is used as context and
a ranking prior, while retrieval still fans out across sparse, dense, heading,
and table-aware signals.

## Install

```bash
pip install sectioniq
```

For local development:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev,bench]"
python -m pytest
```

## Quick Start

```python
from sectioniq import SectionIQ

engine = SectionIQ(store_path=".sectioniq")
doc_id = engine.ingest("/path/to/public-manual.pdf")
engine.build_index()

hits = engine.search("What safety cautions apply before maintenance?", top_k=5)
for hit in hits:
    print(hit.block_type, hit.citation, hit.text_preview)

result = engine.answer("What safety cautions apply before maintenance?", top_k=5)
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

## Public Benchmark Corpus

SectionIQ uses the public-domain U.S. Army `TM-1-1500-204-23` aviation
maintenance manual series as its release validation corpus. The tracked manifest
contains source URLs and metadata only; downloaded PDFs stay local and ignored.

```bash
python scripts/prepare_public_corpus.py
python scripts/benchmark_vs_pageindex.py --rebuild-index
```

To include the optional PageIndex comparison:

```bash
python scripts/benchmark_vs_pageindex.py --run-pageindex
```

See [docs/benchmarking.md](docs/benchmarking.md) for the benchmark workflow.

## Privacy

SectionIQ stores extracted PDF text, metadata, and indexes in the configured
local store. Do not commit local stores, PDFs, notebooks, spreadsheets, logs, or
benchmark outputs from private documents.

## Project Layout

- `src/sectioniq/`: library code
- `benchmarks/`: public corpus manifest and public query set
- `tests/`: unit tests
- `examples/`: local example runners
- `scripts/`: corpus preparation, benchmark, and release-safety utilities
