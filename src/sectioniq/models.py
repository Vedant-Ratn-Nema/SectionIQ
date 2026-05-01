from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    doc_id: str
    source_path: str
    title: str
    page_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    extraction_flags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        return cls(**data)


@dataclass
class Block:
    block_id: str
    doc_id: str
    page_start: int
    page_end: int
    block_type: str
    text: str
    parent_id: str | None = None
    section_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Block":
        if data.get("block_type") == "table":
            return TableBlock.from_dict(data)
        return cls(**data)


@dataclass
class TableBlock(Block):
    rows: list[list[str]] = field(default_factory=list)
    cells: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TableBlock":
        return cls(**data)


@dataclass
class RetrievalHit:
    block_id: str
    doc_id: str
    page_start: int
    page_end: int
    block_type: str
    text_preview: str
    section_path: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    fused_score: float = 0.0
    rerank_score: float = 0.0
    citation: str = ""
    matched_terms: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerEvidence:
    block_id: str
    doc_id: str
    page_start: int
    page_end: int
    block_type: str
    text: str
    citation: str
    matched_terms: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerResult:
    query: str
    answer: str
    evidence: list[AnswerEvidence] = field(default_factory=list)
    hits: list[RetrievalHit] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryBundle:
    raw_query: str
    sparse_query: str
    dense_query: str
    rerank_query: str
    answer_query: str
    extracted_terms: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
