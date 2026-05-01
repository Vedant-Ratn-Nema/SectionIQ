from __future__ import annotations

from pathlib import Path
import os
from typing import Any

from .backends import (
    AnswerGenerator,
    EmbeddingBackend,
    HashEmbeddingBackend,
    HeuristicReranker,
    LocalVectorBackend,
    OpenAIAnswerGenerator,
    OpenAIEmbeddingBackend,
    OpenAIReranker,
    Reranker,
    VectorBackend,
)
from .indexing import BM25Index
from .ingest import IngestionPipeline
from .models import AnswerResult, Block, Document, RetrievalHit
from .preprocess import QueryPreprocessor
from .retrieval import RetrievalEngine
from .storage import LocalStore
from .utils import format_citation


def _env_value(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


class SectionIQ:
    def __init__(
        self,
        store_path: str | Path = ".sectioniq",
        embedding_backend: EmbeddingBackend | None = None,
        vector_backend: VectorBackend | None = None,
        reranker: Reranker | None = None,
        answer_generator: AnswerGenerator | None = None,
        query_preprocessor: QueryPreprocessor | None = None,
    ):
        self.store = LocalStore(store_path)
        self.embedding_backend = embedding_backend or self._default_embedding_backend()
        self.vector_backend = vector_backend or LocalVectorBackend()
        self.reranker = reranker or self._default_reranker()
        self.answer_generator = answer_generator or self._default_answer_generator()
        self.query_preprocessor = query_preprocessor or QueryPreprocessor()
        self.ingestion_pipeline = IngestionPipeline()
        self._embedding_backends: dict[str, EmbeddingBackend] = {self.embedding_backend.name: self.embedding_backend}
        self._vector_backends: dict[str, VectorBackend] = {self.vector_backend.name: self.vector_backend}
        self._rerankers: dict[str, Reranker] = {self.reranker.name: self.reranker}
        self._answer_generators: dict[str, AnswerGenerator] = (
            {self.answer_generator.name: self.answer_generator} if self.answer_generator is not None else {}
        )
        self._bm25_index: BM25Index | None = None
        self._heading_index: BM25Index | None = None
        self._index_mismatch_reason: str | None = None
        self._load_indexes_if_present()

    def register_embedding_backend(self, name: str, backend: EmbeddingBackend) -> None:
        self._embedding_backends[name] = backend

    def register_vector_backend(self, name: str, backend: VectorBackend) -> None:
        self._vector_backends[name] = backend

    def register_reranker(self, name: str, reranker: Reranker) -> None:
        self._rerankers[name] = reranker

    def register_answer_generator(self, name: str, answer_generator: AnswerGenerator) -> None:
        self._answer_generators[name] = answer_generator

    def ingest(
        self,
        path: str,
        metadata: dict[str, Any] | None = None,
        max_pages: int | None = None,
        page_range: tuple[int, int] | None = None,
    ) -> str:
        document, blocks = self.ingestion_pipeline.ingest_file(
            path,
            metadata=metadata,
            max_pages=max_pages,
            page_range=page_range,
        )
        self.store.save_document(document)
        self.store.save_blocks(document.doc_id, blocks)
        return document.doc_id

    def ingest_many(
        self,
        paths: list[str],
        metadata: dict[str, Any] | None = None,
        max_pages: int | None = None,
        page_range: tuple[int, int] | None = None,
    ) -> list[str]:
        return [self.ingest(path, metadata=metadata, max_pages=max_pages, page_range=page_range) for path in paths]

    def build_index(self, doc_ids: list[str] | None = None, backend: str | VectorBackend = "local") -> dict[str, Any]:
        blocks = self._retrievable_blocks(doc_ids)
        if not blocks:
            raise ValueError("No blocks available to index.")

        self._bm25_index = BM25Index()
        self._bm25_index.build(blocks, text_getter=self._block_search_text)

        self._heading_index = BM25Index()
        self._heading_index.build(blocks, text_getter=lambda block: " ".join(block.section_path))

        vectors = self.embedding_backend.embed_texts([self._block_search_text(block) for block in blocks])
        vector_backend = self._resolve_vector_backend(backend)
        vector_backend.build(
            block_ids=[block.block_id for block in blocks],
            vectors=vectors,
            metadata=[{"doc_id": block.doc_id, "block_type": block.block_type} for block in blocks],
        )
        if vector_backend is self.vector_backend:
            self.vector_backend = vector_backend

        self.store.save_payload("bm25.json", self._bm25_index.to_dict())
        self.store.save_payload("heading_bm25.json", self._heading_index.to_dict())
        vector_backend.save(self.store.index_dir / "vectors.npz")
        payload = {
            "vector_backend": vector_backend.name,
            "embedding_backend": self.embedding_backend.name,
            "document_count": len({block.doc_id for block in blocks}),
            "block_count": len(blocks),
        }
        self.store.save_index_metadata(payload)
        return payload

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        self._ensure_indexes_loaded()
        blocks = {block.block_id: block for block in self.store.all_blocks()}
        documents = {doc.doc_id: doc for doc in self.store.list_documents()}
        engine = RetrievalEngine(
            embedding_backend=self.embedding_backend,
            vector_backend=self.vector_backend,
            reranker=self.reranker,
            answer_generator=self.answer_generator,
            query_preprocessor=self.query_preprocessor,
        )
        return engine.search(
            query=query,
            documents=documents,
            blocks=blocks,
            bm25_index=self._bm25_index or BM25Index(),
            heading_index=self._heading_index or BM25Index(),
            top_k=top_k,
            filters=filters,
        )

    def answer(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> AnswerResult:
        hits = self.search(query, top_k=top_k, filters=filters)
        blocks = {block.block_id: block for block in self.store.all_blocks()}
        documents = {doc.doc_id: doc for doc in self.store.list_documents()}
        engine = RetrievalEngine(
            embedding_backend=self.embedding_backend,
            vector_backend=self.vector_backend,
            reranker=self.reranker,
            answer_generator=self.answer_generator,
            query_preprocessor=self.query_preprocessor,
        )
        return engine.answer(query=query, hits=hits, documents=documents, blocks=blocks, top_k=top_k)

    def get_document(self, doc_id: str) -> Document:
        return self.store.load_document(doc_id)

    def get_block(self, block_id: str) -> Block | None:
        return self.store.load_block(block_id)

    def get_citations(self, answer_result: AnswerResult) -> list[str]:
        return [evidence.citation for evidence in answer_result.evidence]

    def export_manifest(self, doc_id: str) -> dict[str, Any]:
        document = self.store.load_document(doc_id)
        blocks = self.store.load_blocks(doc_id)
        return {
            "document": document.to_dict(),
            "blocks": [block.to_dict() for block in blocks],
        }

    def _load_indexes_if_present(self) -> None:
        bm25_path = self.store.index_dir / "bm25.json"
        heading_path = self.store.index_dir / "heading_bm25.json"
        vector_path = self.store.index_dir / "vectors.npz"
        manifest = self.store.load_index_metadata()
        stored_embedding_backend = manifest.get("embedding_backend")
        if stored_embedding_backend and stored_embedding_backend != self.embedding_backend.name:
            self._index_mismatch_reason = (
                "Stored index was built with embedding backend "
                f"'{stored_embedding_backend}', but current engine is using "
                f"'{self.embedding_backend.name}'. Rebuild the index with engine.build_index()."
            )
            if bm25_path.exists():
                self._bm25_index = BM25Index.from_dict(self.store.load_payload("bm25.json"))
            if heading_path.exists():
                self._heading_index = BM25Index.from_dict(self.store.load_payload("heading_bm25.json"))
            return
        if bm25_path.exists():
            self._bm25_index = BM25Index.from_dict(self.store.load_payload("bm25.json"))
        if heading_path.exists():
            self._heading_index = BM25Index.from_dict(self.store.load_payload("heading_bm25.json"))
        if vector_path.exists():
            self.vector_backend.load(vector_path)

    def _ensure_indexes_loaded(self) -> None:
        if self._bm25_index is None or self._heading_index is None:
            self._load_indexes_if_present()
        if self._bm25_index is None or self._heading_index is None:
            raise ValueError("Indexes have not been built yet. Call build_index() first.")
        if self._index_mismatch_reason:
            raise ValueError(self._index_mismatch_reason)

    def _resolve_vector_backend(self, backend: str | VectorBackend) -> VectorBackend:
        if isinstance(backend, str):
            if backend == "local":
                local = self._vector_backends.get("local")
                if local is None:
                    local = LocalVectorBackend()
                    self._vector_backends["local"] = local
                return local
            if backend not in self._vector_backends:
                raise KeyError(f"Unknown vector backend: {backend}")
            return self._vector_backends[backend]
        return backend

    def _retrievable_blocks(self, doc_ids: list[str] | None = None) -> list[Block]:
        allowed = {
            "section",
            "paragraph",
            "list_item",
            "table",
            "figure_caption",
            "warning",
            "note",
        }
        return [block for block in self.store.all_blocks(doc_ids=doc_ids) if block.block_type in allowed and block.text.strip()]

    def _block_search_text(self, block: Block) -> str:
        parts = []
        if block.section_path:
            parts.append(" > ".join(block.section_path))
        parts.append(block.text)
        if block.metadata.get("neighboring_headings", {}).get("parent"):
            parts.append(str(block.metadata["neighboring_headings"]["parent"]))
        return "\n".join(part for part in parts if part)

    def _default_embedding_backend(self) -> EmbeddingBackend:
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIEmbeddingBackend(
                model=_env_value(
                    "SECTIONIQ_EMBEDDING_MODEL",
                    "STRUCTURED_PDF_RAG_EMBEDDING_MODEL",
                    default="text-embedding-3-small",
                )
            )
        return HashEmbeddingBackend()

    def _default_reranker(self) -> Reranker:
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIReranker(
                model=_env_value(
                    "SECTIONIQ_RERANK_MODEL",
                    "SECTIONIQ_LLM_MODEL",
                    "STRUCTURED_PDF_RAG_RERANK_MODEL",
                    "STRUCTURED_PDF_RAG_LLM_MODEL",
                    default="gpt-4o-mini",
                )
            )
        return HeuristicReranker()

    def _default_answer_generator(self) -> AnswerGenerator | None:
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIAnswerGenerator(
                model=_env_value(
                    "SECTIONIQ_LLM_MODEL",
                    "STRUCTURED_PDF_RAG_LLM_MODEL",
                    default="gpt-4o-mini",
                )
            )
        return None


StructuredPDFRAG = SectionIQ
