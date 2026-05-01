from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from .models import AnswerEvidence
from .utils import (
    normalize_whitespace,
    tokenize,
    truncate_text_by_char_budget,
    truncate_text_by_token_count,
)


class EmbeddingBackend(ABC):
    name = "base"

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


class HashEmbeddingBackend(EmbeddingBackend):
    name = "hash"

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                bucket = hash(token) % self.dimension
                matrix[row, bucket] += 1.0
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix


class OpenAIEmbeddingBackend(EmbeddingBackend):
    name = "openai"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 64,
        max_input_tokens: int = 6000,
        max_input_chars: int = 24000,
    ):
        self.model = model
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens
        self.max_input_chars = max_input_chars
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            raw_chunk = [text or " " for text in texts[start : start + self.batch_size]]
            prepared_chunk = [self._prepare_embedding_input(text, self.max_input_tokens, self.max_input_chars) for text in raw_chunk]
            response = self._create_embeddings_with_retry(raw_chunk, prepared_chunk)
            vectors.extend(item.embedding for item in response.data)
        matrix = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _prepare_embedding_input(self, text: str, token_budget: int, char_budget: int) -> str:
        normalized = normalize_whitespace(text or " ")
        compressed = truncate_text_by_char_budget(
            truncate_text_by_token_count(normalized, token_budget),
            char_budget,
        ).strip()
        return compressed or " "

    def _create_embeddings_with_retry(self, raw_chunk: list[str], prepared_chunk: list[str]) -> Any:
        last_error: Exception | None = None
        for token_budget, char_budget in self._retry_budgets():
            current_chunk = [
                self._prepare_embedding_input(text, token_budget, char_budget)
                for text in raw_chunk
            ]
            try:
                return self.client.embeddings.create(model=self.model, input=current_chunk)
            except Exception as exc:
                last_error = exc
                if not self._is_input_too_long_error(exc):
                    raise
        if last_error is not None:
            raise last_error
        return self.client.embeddings.create(model=self.model, input=prepared_chunk)

    def _retry_budgets(self) -> list[tuple[int, int]]:
        budgets = [
            (self.max_input_tokens, self.max_input_chars),
            (min(self.max_input_tokens, 4000), min(self.max_input_chars, 12000)),
            (min(self.max_input_tokens, 2500), min(self.max_input_chars, 8000)),
            (min(self.max_input_tokens, 1500), min(self.max_input_chars, 4000)),
            (min(self.max_input_tokens, 800), min(self.max_input_chars, 2000)),
        ]
        deduped: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for token_budget, char_budget in budgets:
            item = (max(token_budget, 128), max(char_budget, 512))
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _is_input_too_long_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "maximum input length" in message or "too many tokens" in message or "context length" in message


class VectorBackend(ABC):
    name = "base"

    @abstractmethod
    def build(self, block_ids: list[str], vectors: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> None:
        raise NotImplementedError


class LocalVectorBackend(VectorBackend):
    name = "local"

    def __init__(self):
        self.block_ids: list[str] = []
        self.vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.metadata: list[dict[str, Any]] = []

    def build(self, block_ids: list[str], vectors: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        self.block_ids = list(block_ids)
        self.vectors = np.array(vectors, dtype=np.float32)
        self.metadata = list(metadata)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        if self.vectors.size == 0:
            return []
        q = np.array(query_vector, dtype=np.float32)
        if self.vectors.ndim != 2 or q.ndim != 1:
            raise ValueError("Vector search state is invalid. Rebuild the index.")
        if self.vectors.shape[1] != q.shape[0]:
            raise ValueError(
                "Vector dimension mismatch between stored index and current embedding backend. "
                f"Stored dimension={self.vectors.shape[1]}, query dimension={q.shape[0]}. "
                "Rebuild the index with engine.build_index() after loading your intended embedding backend."
            )
        q_norm = np.linalg.norm(q)
        if q_norm:
            q = q / q_norm
        scores = self.vectors @ q
        ranked = np.argsort(scores)[::-1]
        results: list[tuple[str, float]] = []
        for idx in ranked:
            meta = self.metadata[idx]
            if filters and not _metadata_matches(meta, filters):
                continue
            results.append((self.block_ids[idx], float(scores[idx])))
            if len(results) >= top_k:
                break
        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            block_ids=np.array(self.block_ids),
            vectors=self.vectors,
            metadata=np.array(self.metadata, dtype=object),
        )

    def load(self, path: Path) -> None:
        data = np.load(path, allow_pickle=True)
        self.block_ids = [str(item) for item in data["block_ids"].tolist()]
        self.vectors = np.array(data["vectors"], dtype=np.float32)
        self.metadata = [dict(item) for item in data["metadata"].tolist()]


class Reranker(ABC):
    name = "base"

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        analysis: dict[str, Any],
    ) -> list[tuple[str, float]]:
        raise NotImplementedError


class AnswerGenerator(ABC):
    name = "base"

    @abstractmethod
    def generate(
        self,
        query: str,
        evidence: list[AnswerEvidence],
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError


class HeuristicReranker(Reranker):
    name = "heuristic"

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        analysis: dict[str, Any],
    ) -> list[tuple[str, float]]:
        query_tokens = set(tokenize(query))
        results: list[tuple[str, float]] = []
        for candidate in candidates:
            block = candidate["block"]
            hit = candidate["hit"]
            block_tokens = tokenize(block.text)
            overlap = len(query_tokens.intersection(block_tokens))
            overlap_score = overlap / max(len(query_tokens), 1)
            type_bonus = 0.0
            type_penalty = 0.0
            if analysis.get("answer_type") == "table_lookup" and block.block_type == "table":
                type_bonus += 0.75
            if analysis.get("answer_type") == "table_lookup" and block.block_type == "section":
                type_penalty += 0.2
            if analysis.get("answer_type") == "warning" and block.block_type in {"warning", "note"}:
                type_bonus += 0.25
            if analysis.get("answer_type") == "procedure" and block.block_type in {"list_item", "paragraph"}:
                type_bonus += 0.15
            structural_bonus = 0.05 if block.section_path else 0.0
            fused_score = hit.fused_score
            rerank = fused_score * 0.55 + overlap_score * 0.3 + type_bonus + structural_bonus - type_penalty
            results.append((block.block_id, rerank))
        results.sort(key=lambda item: item[1], reverse=True)
        return results


class OpenAIChatClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        content = response.choices[0].message.content or "{}"
        return _extract_json_object(content)


class OpenAIReranker(Reranker):
    name = "openai_llm"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_candidates: int = 12,
    ):
        self.client = OpenAIChatClient(api_key=api_key, model=model, temperature=0.0)
        self.max_candidates = max_candidates

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        analysis: dict[str, Any],
    ) -> list[tuple[str, float]]:
        limited = candidates[: self.max_candidates]
        payload = []
        for idx, candidate in enumerate(limited, start=1):
            block = candidate["block"]
            hit = candidate["hit"]
            payload.append(
                {
                    "rank_id": idx,
                    "block_id": block.block_id,
                    "block_type": block.block_type,
                    "section_path": block.section_path,
                    "citation": hit.citation,
                    "fused_score": hit.fused_score,
                    "text_preview": normalize_whitespace(block.text)[:900],
                }
            )
        system_prompt = (
            "You are a retrieval reranker. Rank candidates by how well they answer the user's query. "
            "Prefer evidence that directly answers the question over loosely related headings. "
            "Return strict JSON only."
        )
        user_prompt = json.dumps(
            {
                "query": query,
                "analysis": analysis,
                "candidates": payload,
                "output_schema": {
                    "ranked_candidates": [
                        {"block_id": "string", "score": "number between 0 and 1"}
                    ]
                },
            },
            ensure_ascii=False,
        )
        parsed = self.client.complete_json(system_prompt, user_prompt)
        ranked = []
        for item in parsed.get("ranked_candidates", []):
            block_id = item.get("block_id")
            score = item.get("score")
            if not isinstance(block_id, str):
                continue
            try:
                ranked.append((block_id, float(score)))
            except (TypeError, ValueError):
                continue
        if ranked:
            return ranked
        heuristic = HeuristicReranker()
        return heuristic.rerank(query, candidates, analysis)


class OpenAIAnswerGenerator(AnswerGenerator):
    name = "openai_llm"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_evidence: int = 8,
    ):
        self.client = OpenAIChatClient(api_key=api_key, model=model, temperature=0.0)
        self.model = model
        self.max_evidence = max_evidence

    def generate(
        self,
        query: str,
        evidence: list[AnswerEvidence],
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        selected = evidence[: self.max_evidence]
        evidence_payload = []
        for item in selected:
            evidence_payload.append(
                {
                    "block_id": item.block_id,
                    "citation": item.citation,
                    "block_type": item.block_type,
                    "page_start": item.page_start,
                    "page_end": item.page_end,
                    "section_path": item.metadata.get("section_path") or item.metadata.get("neighboring_headings") or {},
                    "text": normalize_whitespace(item.text)[:1800],
                }
            )
        system_prompt = (
            "You are a grounded PDF question-answering assistant. "
            "Answer only from the supplied evidence. If the evidence is insufficient, say so plainly. "
            "Write a natural language answer, concise but complete, and choose the evidence block IDs that directly support it. "
            "Return strict JSON only."
        )
        user_prompt = json.dumps(
            {
                "query": query,
                "analysis": analysis,
                "evidence": evidence_payload,
                "requirements": [
                    "Use only provided evidence",
                    "Do not invent facts",
                    "Prefer the most direct evidence",
                    "Return supporting block_ids",
                ],
                "output_schema": {
                    "answer": "string",
                    "supporting_block_ids": ["string"],
                    "confidence": "number between 0 and 1",
                },
            },
            ensure_ascii=False,
        )
        parsed = self.client.complete_json(system_prompt, user_prompt)
        answer = parsed.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("LLM answer generator returned no answer text.")
        block_ids = [item for item in parsed.get("supporting_block_ids", []) if isinstance(item, str)]
        confidence = parsed.get("confidence")
        metadata: dict[str, Any] = {"model": self.model}
        try:
            metadata["confidence"] = float(confidence)
        except (TypeError, ValueError):
            pass
        return {
            "answer": answer.strip(),
            "supporting_block_ids": block_ids,
            "metadata": metadata,
        }


def _metadata_matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    doc_ids = filters.get("doc_ids")
    if doc_ids and metadata.get("doc_id") not in set(doc_ids):
        return False
    block_types = filters.get("block_types")
    if block_types and metadata.get("block_type") not in set(block_types):
        return False
    return True


def _extract_json_object(text: str) -> dict[str, Any]:
    content = (text or "").strip()
    if not content:
        return {}
    if "```json" in content:
        start = content.find("```json") + 7
        end = content.rfind("```")
        content = content[start:end].strip()
    elif "```" in content:
        start = content.find("```") + 3
        end = content.rfind("```")
        content = content[start:end].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return {}
        return {}
