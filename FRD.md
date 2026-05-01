# Functional Requirements Document

## Product Name

SectionIQ

## Objective

Build a Python library that ingests long PDFs into a structured document graph and retrieves evidence using a hybrid retrieval pipeline instead of relying on a single tree path or plain fixed-size chunks.

## Target Users

- Engineers evaluating PDF retrieval quality
- Teams working with manuals, maintenance documents, reports, and technical PDFs
- Developers who want a local-first SDK before deciding whether to add a service layer

## Core Functional Requirements

### 1. Ingestion

The library shall:

- accept a PDF path via `ingest(path, ...)`
- parse page text from digital PDFs
- optionally limit ingest scope with:
  - `max_pages`
  - `page_range=(start, end)`
- normalize extracted text into typed blocks
- preserve source PDF path instead of duplicating the original file

### 2. Document Graph

The library shall create and persist:

- `Document`
  - `doc_id`
  - `source_path`
  - `title`
  - `page_count`
  - `metadata`
  - `extraction_flags`
- `Block`
  - `block_id`
  - `doc_id`
  - `page_start`
  - `page_end`
  - `block_type`
  - `text`
  - `parent_id`
  - `section_path`
  - `metadata`
- `TableBlock`
  - all block fields
  - parsed `rows`
  - parsed `cells`

### 3. Block Types

The library shall classify blocks into:

- `section`
- `paragraph`
- `list_item`
- `table`
- `figure_caption`
- `warning`
- `note`

### 4. Indexing

The library shall support:

- local sparse retrieval using BM25
- local dense retrieval using an embedded vector backend
- metadata-aware retrieval via heading/section path text
- pluggable vector backends for external storage/search

The library shall expose `build_index(doc_ids=None, backend="local"|custom)`.

### 5. Retrieval

The library shall expose `search(query, top_k=..., filters=...)`.

The retrieval pipeline shall:

1. analyze query intent
2. retrieve candidates in parallel from multiple signals:
   - BM25 over block text
   - dense vector search
   - heading/metadata retrieval
   - table-priority retrieval for spec/numeric queries
3. fuse ranked candidates
4. rerank candidates
5. return typed hits with scores and citations

### 6. Answer Generation

The library shall expose `answer(query, top_k=..., filters=...)`.

The answer pipeline shall:

1. take the ranked hits
2. assemble supporting evidence
3. expand context with parent/sibling structural context when useful
4. return a grounded answer using retrieved evidence only
5. include citations

The library shall expose `get_citations(answer_result)`.

### 7. Persistence

The library shall persist:

- document metadata
- block manifests
- BM25 indexes
- vector indexes

The default persistence model shall be local-first.

## Current Pipeline

### Step 1. PDF Parse

`SectionIQ.ingest()` calls the ingestion pipeline, which uses `pypdf` to read pages and metadata.

### Step 2. Structural Heuristics

The ingestion layer scans line-by-line and tries to detect:

- headings
- tables
- list items
- warnings/notes
- paragraphs

### Step 3. Block Graph Build

Each detected unit becomes a block with:

- page span
- parent relationship
- section path
- neighboring heading metadata

### Step 4. Index Build

`build_index()` creates:

- a BM25 index over block text plus section context
- a BM25 index over section paths only
- a dense vector index using the active embedding backend

### Step 5. Query Analysis

Before retrieval, the query is categorized into a rough answer type such as:

- prose
- table lookup
- warning
- procedure
- comparison

### Step 6. Candidate Retrieval

The search layer pulls candidates from multiple channels in parallel and avoids depending on a single branch decision.

### Step 7. Fusion and Reranking

Candidates are merged with rank fusion and then reranked with a heuristic reranker. This is where table queries are intentionally biased toward table blocks instead of heading-only matches.

### Step 8. Context Assembly

The top hit is expanded with:

- parent heading context
- a nearby sibling block when helpful

This gives answer generation both the exact evidence block and surrounding structure.

### Step 9. Grounded Answer

The answer layer synthesizes a short evidence-grounded response and returns citations.

## Public SDK

- `ingest(path, metadata=None, max_pages=None, page_range=None)`
- `ingest_many(paths, metadata=None, max_pages=None, page_range=None)`
- `build_index(doc_ids=None, backend="local"|custom)`
- `search(query, top_k=..., filters=...)`
- `answer(query, top_k=..., filters=...)`
- `get_document(doc_id)`
- `get_block(block_id)`
- `get_citations(answer_result)`
- `register_embedding_backend(name, backend)`
- `register_vector_backend(name, backend)`
- `register_reranker(name, reranker)`
- `export_manifest(doc_id)`

## What Is Implemented Today

- local-first storage
- PDF parsing for digital PDFs
- typed block extraction
- BM25 retrieval
- local dense retrieval or OpenAI embeddings when configured
- heading-aware retrieval
- rank fusion
- heuristic reranking or OpenAI LLM reranking when configured
- LLM-only answer generation when configured
- evidence and citations
- benchmark harness
- example runner for long manuals

## Known V1 Limits

- OCR-heavy scanned PDFs are not handled yet
- heading/table detection is heuristic, not layout-model based
- dense retrieval uses a deterministic hash embedding backend by default
- answer synthesis is intentionally conservative and simple
- benchmarking infrastructure exists, but dataset creation is still manual

## Public Release Corpus

SectionIQ release validation uses the public-domain U.S. Army `TM-1-1500-204-23`
aviation maintenance manual series. The tracked corpus manifest stores source
metadata only; downloaded PDFs, extracted stores, benchmark outputs, and local
PageIndex workspaces remain ignored.

## Recommended Test Flow For Aircraft Manuals

1. run a page-limited ingest first:
   - first 50 pages or a targeted page range
2. inspect block counts and query hit types
3. run 5-10 manual-specific queries:
   - warnings
   - torque/spec values
   - test procedures
   - inspections
   - connector pinouts / tables
4. inspect citations
5. only then run full-document indexing
