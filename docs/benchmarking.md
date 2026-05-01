# Benchmarking

SectionIQ's public release benchmark uses the U.S. Army `TM-1-1500-204-23`
technical manual series from Wikimedia Commons. The PDFs are public-domain U.S.
government works, but they are still downloaded locally and ignored by git.

## Prepare the Corpus

```bash
python scripts/prepare_public_corpus.py
```

This downloads the PDFs listed in `benchmarks/public_corpus_manifest.json` into
`data/public/tm-1-1500-204-23/` and verifies page counts.

## Run SectionIQ

```bash
python scripts/benchmark_vs_pageindex.py --rebuild-index
```

The benchmark reads `benchmarks/public_tm_queries.jsonl`, builds or reuses a
local SectionIQ store, and writes results under `benchmark-results/`.

## Optional PageIndex Comparison

```bash
python scripts/benchmark_vs_pageindex.py --run-pageindex
```

This requires PageIndex and `OPENAI_API_KEY`. PageIndex output is used for
source-level comparison only; raw PDF text and generated outputs remain local.

## Release Gate

Before publishing an alpha release:

- Run unit tests.
- Run the confidential reference scan.
- Run the 20-case public benchmark.
- Publish only aggregate/sanitized benchmark results.
