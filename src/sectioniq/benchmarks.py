from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Block
from .sdk import SectionIQ


@dataclass
class BenchmarkExample:
    query: str
    expected_block_ids: list[str]
    expected_doc_ids: list[str]


class BenchmarkHarness:
    def __init__(self, engine: SectionIQ):
        self.engine = engine

    def load_dataset(self, path: str | Path) -> list[BenchmarkExample]:
        items: list[BenchmarkExample] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                items.append(
                    BenchmarkExample(
                        query=payload["query"],
                        expected_block_ids=payload.get("expected_block_ids", []),
                        expected_doc_ids=payload.get("expected_doc_ids", []),
                    )
                )
        return items

    def evaluate(self, dataset_path: str | Path, top_k: int = 5) -> dict[str, Any]:
        examples = self.load_dataset(dataset_path)
        hybrid_block_hits = 0
        hybrid_doc_hits = 0
        total = len(examples)

        for example in examples:
            hits = self.engine.search(example.query, top_k=top_k)
            hit_block_ids = [hit.block_id for hit in hits]
            hit_doc_ids = [hit.doc_id for hit in hits]
            if example.expected_block_ids and any(block_id in hit_block_ids for block_id in example.expected_block_ids):
                hybrid_block_hits += 1
            if example.expected_doc_ids and any(doc_id in hit_doc_ids for doc_id in example.expected_doc_ids):
                hybrid_doc_hits += 1

        metrics = {
            "examples": total,
            "hybrid_recall_at_k": round(hybrid_block_hits / max(total, 1), 4),
            "hybrid_doc_recall_at_k": round(hybrid_doc_hits / max(total, 1), 4),
        }
        metrics["chunk_baseline_recall_at_k"] = self._chunk_baseline(examples, top_k=top_k)
        metrics["hierarchy_baseline_recall_at_k"] = self._hierarchy_baseline(examples, top_k=top_k)
        return metrics

    def _chunk_baseline(self, examples: list[BenchmarkExample], top_k: int) -> float:
        all_blocks = self.engine.store.all_blocks()
        chunks = self._build_chunks(all_blocks, chunk_chars=400)
        hits = 0
        for example in examples:
            ranked = sorted(chunks, key=lambda block: self._overlap_score(example.query, block.text), reverse=True)[:top_k]
            if any(
                block.metadata.get("source_block_id") in set(example.expected_block_ids)
                for block in ranked
            ):
                hits += 1
        return round(hits / max(len(examples), 1), 4)

    def _hierarchy_baseline(self, examples: list[BenchmarkExample], top_k: int) -> float:
        all_blocks = self.engine.store.all_blocks()
        sections = [block for block in all_blocks if block.block_type == "section"]
        hits = 0
        for example in examples:
            ranked_sections = sorted(
                sections,
                key=lambda block: self._overlap_score(example.query, " ".join(block.section_path + [block.text])),
                reverse=True,
            )[: max(top_k, 1)]
            chosen_ids = {section.block_id for section in ranked_sections}
            descendants = [
                block for block in all_blocks if block.parent_id in chosen_ids or block.block_id in chosen_ids
            ][:top_k]
            if any(block.block_id in set(example.expected_block_ids) for block in descendants):
                hits += 1
        return round(hits / max(len(examples), 1), 4)

    def _build_chunks(self, blocks: list[Block], chunk_chars: int) -> list[Block]:
        chunks: list[Block] = []
        for block in blocks:
            if not block.text.strip():
                continue
            text = block.text
            if len(text) <= chunk_chars:
                chunks.append(
                    Block(
                        block_id=f"{block.block_id}:chunk0",
                        doc_id=block.doc_id,
                        page_start=block.page_start,
                        page_end=block.page_end,
                        block_type="chunk",
                        text=text,
                        parent_id=block.parent_id,
                        section_path=block.section_path,
                        metadata={"source_block_id": block.block_id},
                    )
                )
                continue
            for idx, start in enumerate(range(0, len(text), chunk_chars)):
                chunks.append(
                    Block(
                        block_id=f"{block.block_id}:chunk{idx}",
                        doc_id=block.doc_id,
                        page_start=block.page_start,
                        page_end=block.page_end,
                        block_type="chunk",
                        text=text[start : start + chunk_chars],
                        parent_id=block.parent_id,
                        section_path=block.section_path,
                        metadata={"source_block_id": block.block_id},
                    )
                )
        return chunks

    def _overlap_score(self, query: str, text: str) -> int:
        query_terms = set(query.lower().split())
        text_terms = set(text.lower().split())
        return len(query_terms.intersection(text_terms))
