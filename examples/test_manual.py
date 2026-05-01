#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sectioniq import SectionIQ


DEFAULT_QUERIES = [
    "What warnings are mentioned for maintenance procedures?",
    "Where are torque specifications discussed?",
    "Which section covers inspections or tests?",
]


def parse_page_range(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    start, end = value.split("-", 1)
    return int(start), int(end)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SectionIQ on an aircraft maintenance manual.")
    parser.add_argument("--pdf", required=True, help="Absolute path to the PDF manual.")
    parser.add_argument("--store", default=".sectioniq_demo_store", help="Local store path.")
    parser.add_argument("--max-pages", type=int, default=None, help="Only ingest the first N pages.")
    parser.add_argument("--page-range", default=None, help="Ingest a specific page range like 200-260.")
    parser.add_argument("--query", action="append", default=[], help="Query to run after indexing. Can be repeated.")
    args = parser.parse_args()

    page_range = parse_page_range(args.page_range)
    engine = SectionIQ(store_path=args.store)
    doc_id = engine.ingest(args.pdf, max_pages=args.max_pages, page_range=page_range)
    index_summary = engine.build_index(doc_ids=[doc_id])
    manifest = engine.export_manifest(doc_id)
    block_counts = Counter(block["block_type"] for block in manifest["blocks"])

    print("=== Document Summary ===")
    print(json.dumps(
        {
            "doc_id": doc_id,
            "document": manifest["document"],
            "block_counts": block_counts,
            "index_summary": index_summary,
        },
        indent=2,
        default=list,
    ))

    queries = args.query or DEFAULT_QUERIES
    for query in queries:
        print(f"\n=== Query: {query} ===")
        hits = engine.search(query, top_k=5, filters={"doc_ids": [doc_id]})
        for hit in hits:
            print(f"- {hit.block_type} | {hit.citation} | {hit.text_preview}")
        try:
            answer = engine.answer(query, top_k=5, filters={"doc_ids": [doc_id]})
        except ValueError as exc:
            print("\nAnswer generation unavailable:")
            print(str(exc))
            print("Set OPENAI_API_KEY or pass a custom answer generator.")
            continue
        print("\nAnswer:")
        print(answer.answer)
        print("Citations:")
        for citation in engine.get_citations(answer):
            print(f"- {citation}")


if __name__ == "__main__":
    main()
