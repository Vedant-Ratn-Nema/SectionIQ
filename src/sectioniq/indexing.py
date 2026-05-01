from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .models import Block
from .utils import tokenize


@dataclass
class BM25Document:
    block_id: str
    tokens: list[str]
    metadata: dict[str, Any]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: list[BM25Document] = []
        self.doc_freqs: dict[str, int] = {}
        self.avgdl = 0.0

    def build(self, blocks: list[Block], text_getter=None) -> None:
        self.documents = []
        self.doc_freqs = {}
        total_length = 0
        for block in blocks:
            text = text_getter(block) if text_getter else block.text
            tokens = tokenize(text)
            self.documents.append(
                BM25Document(
                    block_id=block.block_id,
                    tokens=tokens,
                    metadata={"doc_id": block.doc_id, "block_type": block.block_type},
                )
            )
            total_length += len(tokens)
            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
        self.avgdl = total_length / max(len(self.documents), 1)

    def search(self, query: str, top_k: int, filters: dict[str, Any] | None = None) -> list[tuple[str, float]]:
        query_tokens = tokenize(query)
        results: list[tuple[str, float]] = []
        doc_count = max(len(self.documents), 1)
        for document in self.documents:
            if filters and not _metadata_matches(document.metadata, filters):
                continue
            score = 0.0
            term_counts = Counter(document.tokens)
            doc_length = len(document.tokens)
            for term in query_tokens:
                if term not in term_counts:
                    continue
                df = self.doc_freqs.get(term, 0)
                idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
                freq = term_counts[term]
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_length / max(self.avgdl, 1))
                score += idf * (numerator / denominator)
            if score > 0:
                results.append((document.block_id, score))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "k1": self.k1,
            "b": self.b,
            "avgdl": self.avgdl,
            "doc_freqs": self.doc_freqs,
            "documents": [
                {"block_id": doc.block_id, "tokens": doc.tokens, "metadata": doc.metadata}
                for doc in self.documents
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BM25Index":
        index = cls(k1=payload.get("k1", 1.5), b=payload.get("b", 0.75))
        index.avgdl = payload.get("avgdl", 0.0)
        index.doc_freqs = dict(payload.get("doc_freqs", {}))
        index.documents = [
            BM25Document(
                block_id=item["block_id"],
                tokens=list(item.get("tokens", [])),
                metadata=dict(item.get("metadata", {})),
            )
            for item in payload.get("documents", [])
        ]
        return index


def _metadata_matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    doc_ids = filters.get("doc_ids")
    if doc_ids and metadata.get("doc_id") not in set(doc_ids):
        return False
    block_types = filters.get("block_types")
    if block_types and metadata.get("block_type") not in set(block_types):
        return False
    return True
