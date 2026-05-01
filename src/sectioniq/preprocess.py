from __future__ import annotations

import re
from dataclasses import dataclass

from .models import QueryBundle
from .utils import STOPWORDS, normalize_whitespace, sentence_split, tokenize, truncate_text_by_char_budget


IDENTIFIER_RE = re.compile(
    r"\b(?:[A-Z]{2,}[A-Z0-9_-]*|\d{2,}(?:-\d{2,})+|[A-Z]*\d+[A-Z0-9_-]*|\d{3}-\d{4}-\d{3})\b"
)
QUESTION_WORDS = {"what", "why", "how", "where", "when", "which", "who", "interchangeable", "replace", "fault"}
LOW_SIGNAL_PATTERNS = [
    re.compile(r"^(hi|hello|good day|thanks|thank you)\b", re.I),
    re.compile(r"\bplease advise\b", re.I),
    re.compile(r"\bkind regards\b", re.I),
]


@dataclass
class QueryPreprocessorConfig:
    sparse_max_chars: int = 12000
    dense_max_chars: int = 5000
    rerank_max_chars: int = 7000
    answer_max_chars: int = 9000
    max_sentences: int = 12
    max_identifiers: int = 20


class QueryPreprocessor:
    """
    Build stage-specific query representations from arbitrarily long raw input.

    The goal is to preserve high-signal context and exact identifiers without
    sending the entire raw string to every model-bound stage.
    """

    def __init__(self, config: QueryPreprocessorConfig | None = None):
        self.config = config or QueryPreprocessorConfig()

    def prepare(self, query: str) -> QueryBundle:
        raw = normalize_whitespace(query)
        identifiers = self._extract_identifiers(raw)
        tokens = [token for token in tokenize(raw) if token not in STOPWORDS]
        selected_sentences = self._select_sentences(raw, identifiers)

        summary_parts: list[str] = []
        if identifiers:
            summary_parts.append("Identifiers: " + ", ".join(identifiers[: self.config.max_identifiers]))
        if selected_sentences:
            summary_parts.append("Key context: " + " ".join(selected_sentences))
        elif raw:
            summary_parts.append(raw)
        summary = "\n".join(summary_parts).strip() or raw

        sparse_query = truncate_text_by_char_budget(self._merge_parts([raw, summary]), self.config.sparse_max_chars)
        dense_query = truncate_text_by_char_budget(summary, self.config.dense_max_chars)
        rerank_query = truncate_text_by_char_budget(self._merge_parts([summary, raw[: self.config.rerank_max_chars]]), self.config.rerank_max_chars)
        answer_query = truncate_text_by_char_budget(self._merge_parts([raw, summary]), self.config.answer_max_chars)

        return QueryBundle(
            raw_query=raw,
            sparse_query=sparse_query,
            dense_query=dense_query,
            rerank_query=rerank_query,
            answer_query=answer_query,
            extracted_terms=identifiers[: self.config.max_identifiers] + tokens[:20],
            metadata={
                "raw_chars": len(raw),
                "sparse_chars": len(sparse_query),
                "dense_chars": len(dense_query),
                "rerank_chars": len(rerank_query),
                "answer_chars": len(answer_query),
                "query_was_compressed": len(raw) > len(dense_query) or len(raw) > len(answer_query),
                "identifier_count": len(identifiers),
                "selected_sentence_count": len(selected_sentences),
            },
        )

    def _extract_identifiers(self, text: str) -> list[str]:
        found = []
        seen = set()
        for match in IDENTIFIER_RE.finditer(text):
            value = match.group(0)
            if value not in seen:
                seen.add(value)
                found.append(value)
        return found

    def _select_sentences(self, text: str, identifiers: list[str]) -> list[str]:
        sentences = sentence_split(text)
        if not sentences:
            return []
        scored: list[tuple[float, int, str]] = []
        identifier_set = {item.lower() for item in identifiers}
        for idx, sentence in enumerate(sentences):
            normalized = normalize_whitespace(sentence)
            if not normalized:
                continue
            if any(pattern.search(normalized) for pattern in LOW_SIGNAL_PATTERNS):
                continue
            sentence_tokens = tokenize(normalized)
            token_set = set(sentence_tokens)
            identifier_hits = sum(1 for ident in identifier_set if ident in normalized.lower())
            question_hits = sum(1 for token in token_set if token in QUESTION_WORDS)
            numeric_hits = sum(1 for token in sentence_tokens if any(ch.isdigit() for ch in token))
            early_bonus = max(0, 3 - idx) * 0.1
            score = identifier_hits * 3.0 + question_hits * 1.5 + numeric_hits * 0.25 + early_bonus + min(len(sentence_tokens), 40) / 100.0
            scored.append((score, idx, normalized))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = []
        seen = set()
        for _, _, sentence in scored:
            lowered = sentence.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            selected.append(sentence)
            if len(selected) >= self.config.max_sentences:
                break
        selected.sort(key=lambda sentence: text.find(sentence))
        return selected

    def _merge_parts(self, parts: list[str]) -> str:
        seen = set()
        result = []
        for part in parts:
            normalized = normalize_whitespace(part)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return "\n".join(result)
