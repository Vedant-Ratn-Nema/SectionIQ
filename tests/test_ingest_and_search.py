from __future__ import annotations

from types import SimpleNamespace

import sectioniq
from sectioniq.backends import AnswerGenerator
from sectioniq.backends import OpenAIEmbeddingBackend
from sectioniq.backends import VectorBackend
from sectioniq.models import AnswerEvidence
from sectioniq.ingest import IngestionPipeline
from sectioniq.indexing import BM25Index
from sectioniq.retrieval import RetrievalEngine
from sectioniq.sdk import SectionIQ
from sectioniq.utils import estimate_provider_safe_token_count


class FakeAnswerGenerator(AnswerGenerator):
    name = "fake"

    def generate(self, query: str, evidence: list[AnswerEvidence], analysis: dict[str, object]) -> dict[str, object]:
        first = evidence[0]
        return {
            "answer": f"Mock answer for: {query}",
            "supporting_block_ids": [first.block_id],
            "metadata": {"model": "fake-llm"},
        }


def build_engine(tmp_path):
    engine = SectionIQ(store_path=tmp_path / "store", answer_generator=FakeAnswerGenerator())
    pipeline = IngestionPipeline()

    manual_doc, manual_blocks = pipeline.ingest_pages(
        source_path=str(tmp_path / "manual_a.pdf"),
        title="Manual A",
        pages=[
            "\n".join(
                [
                    "INTRODUCTION",
                    "This manual covers installation details.",
                    "Torque Specifications",
                    "Fastener    Torque    Unit",
                    "Bolt A      30        Nm",
                    "Bolt B      45        Nm",
                    "Warning: Disconnect power before servicing.",
                ]
            )
        ],
    )
    engine.store.save_document(manual_doc)
    engine.store.save_blocks(manual_doc.doc_id, manual_blocks)

    report_doc, report_blocks = pipeline.ingest_pages(
        source_path=str(tmp_path / "report_b.pdf"),
        title="Report B",
        pages=[
            "\n".join(
                [
                    "Quarterly Overview",
                    "Revenue increased 15 percent year over year.",
                    "Risk Factors",
                    "Supply chain volatility remains a concern.",
                ]
            )
        ],
    )
    engine.store.save_document(report_doc)
    engine.store.save_blocks(report_doc.doc_id, report_blocks)
    engine.build_index()
    return engine, manual_doc.doc_id, report_doc.doc_id


def test_search_prefers_table_for_spec_queries(tmp_path):
    engine, manual_doc_id, _ = build_engine(tmp_path)
    hits = engine.search("What is the torque for Bolt B?", top_k=3, filters={"doc_ids": [manual_doc_id]})
    assert hits
    assert hits[0].block_type == "table"


def test_search_can_retrieve_across_documents(tmp_path):
    engine, _, report_doc_id = build_engine(tmp_path)
    hits = engine.search("What happened to revenue?", top_k=3)
    assert hits
    assert hits[0].doc_id == report_doc_id


def test_answer_returns_citations(tmp_path):
    engine, manual_doc_id, _ = build_engine(tmp_path)
    result = engine.answer("What is the torque for Bolt A?", top_k=3, filters={"doc_ids": [manual_doc_id]})
    citations = engine.get_citations(result)
    assert citations
    assert "Manual A" in citations[0]
    assert result.metadata["model"] == "fake-llm"


def test_export_manifest_contains_document_and_blocks(tmp_path):
    engine, manual_doc_id, _ = build_engine(tmp_path)
    manifest = engine.export_manifest(manual_doc_id)
    assert manifest["document"]["doc_id"] == manual_doc_id
    assert manifest["blocks"]


def test_openai_embedding_backend_retries_with_smaller_payloads():
    class FakeEmbeddingsAPI:
        def __init__(self):
            self.calls = []

        def create(self, model: str, input: list[str]):
            self.calls.append(input)
            if any(estimate_provider_safe_token_count(text) > 1000 for text in input):
                raise Exception("Invalid 'input[0]': maximum input length is 8192 tokens.")
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[1.0, 0.0, 0.0]) for _ in input]
            )

    backend = OpenAIEmbeddingBackend(api_key="test-key", max_input_tokens=6000, max_input_chars=24000)
    fake_embeddings = FakeEmbeddingsAPI()
    backend.client = SimpleNamespace(embeddings=fake_embeddings)

    huge_query = ("Part ABC-1234-567 needs review. " + ("x ! " * 7000)).strip()
    vectors = backend.embed_texts([huge_query])

    assert vectors.shape == (1, 3)
    assert len(fake_embeddings.calls) >= 2
    assert estimate_provider_safe_token_count(fake_embeddings.calls[-1][0]) <= 1000


def test_search_falls_back_when_dense_query_embedding_fails(tmp_path):
    engine, manual_doc_id, _ = build_engine(tmp_path)
    blocks = {block.block_id: block for block in engine.store.all_blocks()}
    documents = {doc.doc_id: doc for doc in engine.store.list_documents()}

    class ExplodingEmbeddingBackend:
        name = "exploding"

        def embed_texts(self, texts):
            raise Exception("Invalid 'input[0]': maximum input length is 8192 tokens.")

        def embed_query(self, text):
            raise Exception("Invalid 'input[0]': maximum input length is 8192 tokens.")

    class EmptyVectorBackend(VectorBackend):
        name = "empty"

        def build(self, block_ids, vectors, metadata):
            return None

        def search(self, query_vector, top_k, filters=None):
            return []

        def save(self, path):
            return None

        def load(self, path):
            return None

    retrieval = RetrievalEngine(
        embedding_backend=ExplodingEmbeddingBackend(),
        vector_backend=EmptyVectorBackend(),
        answer_generator=FakeAnswerGenerator(),
    )
    hits = retrieval.search(
        query="What is the torque for Bolt B?",
        documents=documents,
        blocks=blocks,
        bm25_index=engine._bm25_index or BM25Index(),
        heading_index=engine._heading_index or BM25Index(),
        top_k=3,
        filters={"doc_ids": [manual_doc_id]},
    )
    assert hits
    assert hits[0].block_type == "table"


def test_collapsed_technical_rows_become_table_blocks(tmp_path):
    pipeline = IngestionPipeline()
    _, blocks = pipeline.ingest_pages(
        source_path=str(tmp_path / "public_manual.pdf"),
        title="Public Manual",
        pages=[
            "\n".join(
                [
                    "PARTS LIST",
                    "12-34-56 ABC-1234-567 Servo Mount 1",
                    "12-34-57 XYZ-0001-222 Support Bracket 2",
                ]
            )
        ],
    )
    tables = [block for block in blocks if block.block_type == "table"]
    assert tables
    assert tables[0].rows[0] == ["12-34-56", "ABC-1234-567", "Servo Mount", "1"]


def test_warning_labels_are_strict(tmp_path):
    pipeline = IngestionPipeline()
    _, blocks = pipeline.ingest_pages(
        source_path=str(tmp_path / "public_manual.pdf"),
        title="Public Manual",
        pages=[
            "\n".join(
                [
                    "WARNING: Disconnect electrical power before maintenance.",
                    "Warning System Test Procedure",
                    "NOTE: Use approved cleaning solvent.",
                ]
            )
        ],
    )
    by_text = {block.text: block.block_type for block in blocks}
    assert by_text["WARNING: Disconnect electrical power before maintenance."] == "warning"
    assert by_text["Warning System Test Procedure"] != "warning"
    assert by_text["NOTE: Use approved cleaning solvent."] == "note"


def test_deprecated_import_namespace_still_resolves():
    from structured_pdf_rag import StructuredPDFRAG
    from structured_pdf_rag.utils import truncate_text_by_char_budget

    assert StructuredPDFRAG is SectionIQ
    assert truncate_text_by_char_budget("abcdef", 3).startswith("abc")
    assert sectioniq.__version__ == "0.1.0a1"


def test_long_query_preprocessing_preserves_identifiers():
    from sectioniq.preprocess import QueryPreprocessor

    bundle = QueryPreprocessor().prepare(("Please review ABC-1234-567. " + "background details " * 1000).strip())
    assert "ABC-1234-567" in bundle.dense_query
    assert "ABC-1234-567" in bundle.rerank_query
    assert "ABC-1234-567" in bundle.answer_query
