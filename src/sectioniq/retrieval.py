from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from .backends import AnswerGenerator, EmbeddingBackend, HeuristicReranker, Reranker, VectorBackend
from .indexing import BM25Index
from .models import AnswerEvidence, AnswerResult, Block, Document, QueryBundle, RetrievalHit, TableBlock
from .preprocess import QueryPreprocessor
from .utils import STOPWORDS, format_citation, normalize_whitespace, preview_text, sentence_split, tokenize


class QueryAnalyzer:
    def analyze(self, query: str) -> dict[str, Any]:
        lowered = query.lower()
        answer_type = "prose"
        if re.match(r"^\s*(what is|what's|identify|find)\s+", lowered):
            answer_type = "entity_lookup"
        if any(term in lowered for term in ["table", "value", "spec", "specification", "voltage", "torque", "rating"]):
            answer_type = "table_lookup"
        elif any(term in lowered for term in ["warning", "caution", "hazard", "danger", "safety"]):
            answer_type = "warning"
        elif any(term in lowered for term in ["steps", "procedure", "how to", "install", "configure", "replace"]):
            answer_type = "procedure"
        elif any(term in lowered for term in ["compare", "difference", "versus", "vs"]):
            answer_type = "comparison"
        subject = query.strip()
        subject = re.sub(r"^\s*(what is|what's|identify|find)\s+", "", subject, flags=re.I)
        return {
            "answer_type": answer_type,
            "tokens": [token for token in tokenize(query) if token not in STOPWORDS],
            "table_priority": answer_type == "table_lookup",
            "subject": normalize_whitespace(subject).rstrip(" ?"),
        }


class RetrievalEngine:
    def __init__(
        self,
        embedding_backend: EmbeddingBackend,
        vector_backend: VectorBackend,
        reranker: Reranker | None = None,
        answer_generator: AnswerGenerator | None = None,
        query_preprocessor: QueryPreprocessor | None = None,
    ):
        self.embedding_backend = embedding_backend
        self.vector_backend = vector_backend
        self.reranker = reranker or HeuristicReranker()
        self.answer_generator = answer_generator
        self.query_analyzer = QueryAnalyzer()
        self.query_preprocessor = query_preprocessor or QueryPreprocessor()

    def search(
        self,
        query: str,
        documents: dict[str, Document],
        blocks: dict[str, Block],
        bm25_index: BM25Index,
        heading_index: BM25Index,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        query_bundle = self.query_preprocessor.prepare(query)
        analysis = self.query_analyzer.analyze(query_bundle.answer_query)
        analysis["query_bundle"] = query_bundle.metadata
        analysis["extracted_terms"] = query_bundle.extracted_terms
        candidate_map: dict[str, RetrievalHit] = {}

        sparse_hits = bm25_index.search(query_bundle.sparse_query, top_k=max(top_k * 4, 10), filters=filters)
        heading_hits = heading_index.search(query_bundle.sparse_query, top_k=max(top_k * 4, 10), filters=filters)
        dense_hits: list[tuple[str, float]] = []
        try:
            dense_hits = self.vector_backend.search(
                self.embedding_backend.embed_query(query_bundle.dense_query),
                top_k=max(top_k * 4, 10),
                filters=filters,
            )
        except Exception as exc:
            analysis["dense_search_error"] = str(exc)
        table_hits = []
        if analysis["table_priority"]:
            table_filters = dict(filters or {})
            block_types = set(table_filters.get("block_types", []))
            block_types.add("table")
            table_filters["block_types"] = list(block_types)
            table_hits = bm25_index.search(query_bundle.sparse_query, top_k=max(top_k * 3, 8), filters=table_filters)

        self._accumulate_ranked_hits(candidate_map, sparse_hits, blocks, documents, "bm25")
        self._accumulate_ranked_hits(candidate_map, dense_hits, blocks, documents, "dense")
        self._accumulate_ranked_hits(candidate_map, heading_hits, blocks, documents, "heading")
        self._accumulate_ranked_hits(candidate_map, table_hits, blocks, documents, "table")
        self._apply_exact_match_boost(candidate_map, analysis, blocks)

        candidates = sorted(candidate_map.values(), key=lambda hit: hit.fused_score, reverse=True)[: max(top_k * 5, 15)]
        reranked = self.reranker.rerank(
            query_bundle.rerank_query,
            [{"hit": hit, "block": blocks[hit.block_id]} for hit in candidates],
            analysis,
        )
        rerank_map = {block_id: score for block_id, score in reranked}
        for hit in candidates:
            hit.rerank_score = rerank_map.get(hit.block_id, hit.fused_score)
        candidates.sort(key=lambda hit: (hit.rerank_score, hit.fused_score), reverse=True)
        return candidates[:top_k]

    def answer(
        self,
        query: str,
        hits: list[RetrievalHit],
        documents: dict[str, Document],
        blocks: dict[str, Block],
        top_k: int = 5,
    ) -> AnswerResult:
        evidence = self._assemble_evidence(hits[:top_k], documents, blocks)
        query_bundle = self.query_preprocessor.prepare(query)
        analysis = self.query_analyzer.analyze(query_bundle.answer_query)
        analysis["query_bundle"] = query_bundle.metadata
        analysis["extracted_terms"] = query_bundle.extracted_terms
        if self.answer_generator is None:
            raise ValueError(
                "No answer generator configured. Provide an LLM-backed answer generator to use answer()."
            )
        generation = self.answer_generator.generate(query=query_bundle.answer_query, evidence=evidence, analysis=analysis)
        answer = generation["answer"]
        selected_ids = set(generation.get("supporting_block_ids", []))
        if selected_ids:
            evidence = [item for item in evidence if item.block_id in selected_ids] or evidence
        return AnswerResult(
            query=query,
            answer=answer,
            evidence=evidence,
            hits=hits[:top_k],
            metadata={"answer_type": analysis["answer_type"], "query_bundle": query_bundle.metadata, **generation.get("metadata", {})},
        )

    def _accumulate_ranked_hits(
        self,
        candidate_map: dict[str, RetrievalHit],
        results: list[tuple[str, float]],
        blocks: dict[str, Block],
        documents: dict[str, Document],
        score_name: str,
    ) -> None:
        for rank, (block_id, score) in enumerate(results, start=1):
            block = blocks.get(block_id)
            if not block:
                continue
            document = documents[block.doc_id]
            hit = candidate_map.get(block_id)
            if not hit:
                hit = RetrievalHit(
                    block_id=block.block_id,
                    doc_id=block.doc_id,
                    page_start=block.page_start,
                    page_end=block.page_end,
                    block_type=block.block_type,
                    text_preview=preview_text(block.text),
                    section_path=list(block.section_path),
                    citation=format_citation(document.title, block.page_start, block.page_end, block.block_id),
                    metadata=block.metadata,
                )
                candidate_map[block_id] = hit
            hit.scores[score_name] = float(score)
            hit.fused_score += 1.0 / (60 + rank)
            hit.matched_terms = sorted(set(hit.matched_terms).union(self._matched_terms(block.text, hit, score_name)))

    def _matched_terms(self, text: str, hit: RetrievalHit, score_name: str) -> list[str]:
        tokens = set(tokenize(text))
        return [term for term in tokenize(hit.text_preview) if term in tokens][:10] if score_name == "heading" else []

    def _apply_exact_match_boost(
        self,
        candidate_map: dict[str, RetrievalHit],
        analysis: dict[str, Any],
        blocks: dict[str, Block],
    ) -> None:
        subject = analysis.get("subject", "")
        if not subject:
            return
        subject_lower = subject.lower()
        subject_tokens = set(tokenize(subject))
        for hit in candidate_map.values():
            block = blocks[hit.block_id]
            haystacks = [block.text.lower(), " > ".join(block.section_path).lower()]
            if any(subject_lower and subject_lower in hay for hay in haystacks):
                hit.scores["exact_match"] = 1.0
                hit.fused_score += 0.1
                continue
            overlap = len(subject_tokens.intersection(tokenize(block.text)))
            if subject_tokens and overlap >= max(2, len(subject_tokens) - 1):
                hit.scores["subject_overlap"] = float(overlap)
                hit.fused_score += 0.03

    def _assemble_evidence(
        self,
        hits: list[RetrievalHit],
        documents: dict[str, Document],
        blocks: dict[str, Block],
    ) -> list[AnswerEvidence]:
        evidence: list[AnswerEvidence] = []
        seen: set[str] = set()
        siblings_by_parent: dict[str | None, list[Block]] = defaultdict(list)
        page_blocks: dict[tuple[str, int], list[Block]] = defaultdict(list)
        for block in blocks.values():
            siblings_by_parent[block.parent_id].append(block)
            page_blocks[(block.doc_id, block.page_start)].append(block)
        for siblings in siblings_by_parent.values():
            siblings.sort(key=lambda block: block.block_id)
        for same_page in page_blocks.values():
            same_page.sort(key=lambda block: block.block_id)

        for hit in hits:
            if hit.block_id in seen:
                continue
            block = blocks[hit.block_id]
            document = documents[block.doc_id]
            self._append_evidence(
                evidence,
                seen,
                document,
                block,
                matched_terms=hit.matched_terms,
                metadata=block.metadata,
            )
            parent = blocks.get(block.parent_id) if block.parent_id else None
            if parent and parent.block_id not in seen:
                self._append_evidence(
                    evidence,
                    seen,
                    document,
                    parent,
                    matched_terms=[],
                    metadata={"context_role": "parent_heading", **parent.metadata},
                )
            sibling_blocks = self._candidate_context_blocks(block, siblings_by_parent, page_blocks)
            for sibling in sibling_blocks[:1]:
                if sibling.block_id in seen:
                    continue
                self._append_evidence(
                    evidence,
                    seen,
                    document,
                    sibling,
                    matched_terms=[],
                    metadata={"context_role": "sibling", **sibling.metadata},
                )
        return evidence

    def _append_evidence(
        self,
        evidence: list[AnswerEvidence],
        seen: set[str],
        document: Document,
        block: Block,
        matched_terms: list[str],
        metadata: dict[str, Any],
    ) -> None:
        text_key = (block.doc_id, block.page_start, normalize_whitespace(block.text))
        if block.block_id in seen or text_key in seen:
            return
        evidence.append(
            AnswerEvidence(
                block_id=block.block_id,
                doc_id=block.doc_id,
                page_start=block.page_start,
                page_end=block.page_end,
                block_type=block.block_type,
                text=block.text,
                citation=format_citation(document.title, block.page_start, block.page_end, block.block_id),
                matched_terms=matched_terms,
                metadata=metadata,
            )
        )
        seen.add(block.block_id)
        seen.add(text_key)

    def _candidate_context_blocks(
        self,
        block: Block,
        siblings_by_parent: dict[str | None, list[Block]],
        page_blocks: dict[tuple[str, int], list[Block]],
    ) -> list[Block]:
        if block.parent_id:
            siblings = [sib for sib in siblings_by_parent.get(block.parent_id, []) if sib.block_id != block.block_id]
            return siblings
        same_page = [sib for sib in page_blocks.get((block.doc_id, block.page_start), []) if sib.block_id != block.block_id]
        same_page = [sib for sib in same_page if sib.block_type != "section" or sib.page_start == block.page_start]
        return same_page
