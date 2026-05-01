#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
VENDOR = ROOT / ".vendor"
PAGEINDEX_ROOT = Path("/tmp/PageIndex")
for import_path in [SRC, VENDOR, PAGEINDEX_ROOT]:
    if import_path.exists() and str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from sectioniq import SectionIQ


DEFAULT_MANIFEST = ROOT / "benchmarks" / "public_corpus_manifest.json"
DEFAULT_QUERIES = ROOT / "benchmarks" / "public_tm_queries.jsonl"


@dataclass
class QueryCase:
    id: str
    category: str
    query: str
    expected_source_ids: list[str]
    expected_terms: list[str]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cases(path: Path, limit: int | None = None) -> list[QueryCase]:
    cases: list[QueryCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            cases.append(
                QueryCase(
                    id=payload["id"],
                    category=payload.get("category", "unknown"),
                    query=payload["query"],
                    expected_source_ids=list(payload.get("expected_source_ids", [])),
                    expected_terms=list(payload.get("expected_terms", [])),
                )
            )
            if limit is not None and len(cases) >= limit:
                break
    return cases


def source_paths(manifest: dict[str, Any]) -> list[tuple[dict[str, Any], Path]]:
    local_dir = ROOT / manifest["local_dir"]
    return [(source, local_dir / source["filename"]) for source in manifest["sources"]]


def ensure_public_pdfs_available(manifest: dict[str, Any]) -> None:
    missing = [str(path) for _, path in source_paths(manifest) if not path.exists()]
    if missing:
        message = [
            "Public corpus PDFs are missing.",
            "Run: python scripts/prepare_public_corpus.py",
            "Missing files:",
            *[f"- {item}" for item in missing],
        ]
        raise FileNotFoundError("\n".join(message))


def build_sectioniq_engine(manifest: dict[str, Any], store_path: str, rebuild_index: bool) -> SectionIQ:
    engine = SectionIQ(store_path=store_path)
    existing = engine.store.list_documents()
    if not existing:
        for source, path in source_paths(manifest):
            engine.ingest(
                str(path),
                metadata={
                    "source_id": source["id"],
                    "source_url": source["source_url"],
                    "public_corpus_id": manifest["corpus_id"],
                },
            )
        rebuild_index = True
    if rebuild_index:
        engine.build_index()
    return engine


def evaluate_sectioniq(engine: SectionIQ, cases: list[QueryCase], top_k: int, run_answers: bool) -> list[dict[str, Any]]:
    docs_by_id = {doc.doc_id: doc for doc in engine.store.list_documents()}
    results: list[dict[str, Any]] = []
    for case in cases:
        started = time.time()
        hits = engine.search(case.query, top_k=top_k)
        hit_source_ids = [
            docs_by_id[hit.doc_id].metadata.get("source_id", hit.doc_id)
            for hit in hits
        ]
        preview_text = " ".join(hit.text_preview.lower() for hit in hits)
        source_match = bool(set(case.expected_source_ids).intersection(hit_source_ids))
        term_match = all(term.lower() in preview_text for term in case.expected_terms)
        item: dict[str, Any] = {
            "id": case.id,
            "category": case.category,
            "query": case.query,
            "expected_source_ids": case.expected_source_ids,
            "sectioniq": {
                "source_match": source_match,
                "term_match": term_match,
                "top_sources": hit_source_ids[:top_k],
                "hits": [
                    {
                        "block_type": hit.block_type,
                        "citation": hit.citation,
                        "text_preview": hit.text_preview,
                        "scores": hit.scores,
                    }
                    for hit in hits
                ],
            },
            "elapsed_seconds": round(time.time() - started, 3),
        }
        if run_answers:
            answer = engine.answer(case.query, top_k=top_k)
            item["sectioniq"]["answer"] = answer.answer
            item["sectioniq"]["citations"] = engine.get_citations(answer)
            item["sectioniq"]["answer_metadata"] = answer.metadata
        results.append(item)
    return results


def tokenize(text: str) -> set[str]:
    return {part.strip(".,:;()[]{}").lower() for part in (text or "").split() if part.strip(".,:;()[]{}")}


def maybe_pageindex_client(model: str, workspace: str):
    try:
        module = importlib.import_module("pageindex")
    except ImportError as exc:
        raise ImportError(
            "PageIndex is not installed. Install it or clone it into /tmp/PageIndex before using --run-pageindex."
        ) from exc
    return module.PageIndexClient(model=model, retrieve_model=model, workspace=workspace)


def ensure_pageindex_indexed(client, manifest: dict[str, Any], reindex: bool = False) -> dict[str, str]:
    source_to_doc: dict[str, str] = {}
    for source, pdf_path in source_paths(manifest):
        resolved = str(pdf_path.resolve())
        if not reindex:
            for doc_id, doc in client.documents.items():
                if doc.get("path") == resolved:
                    source_to_doc[source["id"]] = doc_id
                    break
        if source["id"] not in source_to_doc:
            source_to_doc[source["id"]] = client.index(resolved)
    return source_to_doc


def flatten_pageindex_nodes(nodes: list[dict[str, Any]], source_id: str) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for node in nodes:
        flat.append(
            {
                "source_id": source_id,
                "title": str(node.get("title", "")),
                "summary": str(node.get("summary", "")),
                "start_index": node.get("start_index"),
                "end_index": node.get("end_index"),
            }
        )
        flat.extend(flatten_pageindex_nodes(node.get("nodes") or [], source_id=source_id))
    return flat


def evaluate_pageindex_sources(client, source_to_doc: dict[str, str], cases: list[QueryCase], top_k: int) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for source_id, doc_id in source_to_doc.items():
        structure = json.loads(client.get_document_structure(doc_id))
        nodes.extend(flatten_pageindex_nodes(structure, source_id=source_id))

    results = []
    for case in cases:
        query_terms = tokenize(case.query)
        ranked = sorted(
            nodes,
            key=lambda node: len(query_terms.intersection(tokenize(node["title"] + " " + node["summary"]))),
            reverse=True,
        )
        top_sources: list[str] = []
        top_nodes = []
        for node in ranked:
            if node["source_id"] not in top_sources:
                top_sources.append(node["source_id"])
            top_nodes.append(node)
            if len(top_nodes) >= top_k:
                break
        results.append(
            {
                "id": case.id,
                "source_match": bool(set(case.expected_source_ids).intersection(top_sources)),
                "top_sources": top_sources[:top_k],
                "top_nodes": top_nodes,
            }
        )
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    source_hits = sum(1 for item in results if item["sectioniq"]["source_match"])
    term_hits = sum(1 for item in results if item["sectioniq"]["term_match"])
    by_category: dict[str, dict[str, Any]] = {}
    for item in results:
        slot = by_category.setdefault(item["category"], {"count": 0, "source_hits": 0, "term_hits": 0})
        slot["count"] += 1
        slot["source_hits"] += int(item["sectioniq"]["source_match"])
        slot["term_hits"] += int(item["sectioniq"]["term_match"])
    for slot in by_category.values():
        slot["source_recall"] = round(slot["source_hits"] / max(slot["count"], 1), 4)
        slot["term_recall"] = round(slot["term_hits"] / max(slot["count"], 1), 4)
    return {
        "cases": total,
        "sectioniq_source_recall": round(source_hits / max(total, 1), 4),
        "sectioniq_term_recall": round(term_hits / max(total, 1), 4),
        "by_category": by_category,
    }


def attach_pageindex_results(payload_results: list[dict[str, Any]], pageindex_results: list[dict[str, Any]]) -> None:
    pageindex_by_id = {item["id"]: item for item in pageindex_results}
    for item in payload_results:
        if item["id"] in pageindex_by_id:
            item["pageindex"] = pageindex_by_id[item["id"]]


def add_pageindex_summary(summary: dict[str, Any], pageindex_results: list[dict[str, Any]]) -> None:
    total = len(pageindex_results)
    hits = sum(1 for item in pageindex_results if item["source_match"])
    summary["pageindex_source_recall"] = round(hits / max(total, 1), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SectionIQ on the public Army TM corpus.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Public corpus manifest.")
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES), help="Public benchmark query JSONL.")
    parser.add_argument("--store", default=".sectioniq_public_store", help="Local SectionIQ store path.")
    parser.add_argument("--output", default="benchmark-results/public_tm_sectioniq.json", help="Output JSON path.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of hits to evaluate.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of query cases.")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild the SectionIQ index.")
    parser.add_argument("--run-answers", action="store_true", help="Run LLM-backed answer generation.")
    parser.add_argument("--run-pageindex", action="store_true", help="Also run PageIndex source-level retrieval comparison.")
    parser.add_argument("--pageindex-workspace", default=".public_pageindex_workspace", help="PageIndex workspace path.")
    parser.add_argument("--pageindex-model", default=os.getenv("SECTIONIQ_LLM_MODEL", "gpt-4o-mini"), help="Model passed to PageIndex.")
    parser.add_argument("--pageindex-reindex", action="store_true", help="Force PageIndex to reindex public PDFs.")
    args = parser.parse_args()

    if args.run_answers and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required when --run-answers is set.")

    manifest = load_json(Path(args.manifest))
    ensure_public_pdfs_available(manifest)
    cases = load_cases(Path(args.queries), limit=args.limit)
    engine = build_sectioniq_engine(manifest, store_path=args.store, rebuild_index=args.rebuild_index)
    results = evaluate_sectioniq(engine, cases, top_k=args.top_k, run_answers=args.run_answers)
    pageindex_results = []
    if args.run_pageindex:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required when --run-pageindex is set.")
        pageindex_client = maybe_pageindex_client(model=args.pageindex_model, workspace=args.pageindex_workspace)
        source_to_doc = ensure_pageindex_indexed(pageindex_client, manifest, reindex=args.pageindex_reindex)
        pageindex_results = evaluate_pageindex_sources(pageindex_client, source_to_doc, cases, top_k=args.top_k)
        attach_pageindex_results(results, pageindex_results)
    summary = summarize(results)
    if pageindex_results:
        add_pageindex_summary(summary, pageindex_results)
    payload = {
        "corpus_id": manifest["corpus_id"],
        "summary": summary,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
