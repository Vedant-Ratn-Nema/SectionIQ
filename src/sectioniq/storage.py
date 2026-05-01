from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import Block, Document
from .utils import dump_json, load_json


class LocalStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.docs_dir = self.root / "documents"
        self.blocks_dir = self.root / "blocks"
        self.index_dir = self.root / "indexes"
        self.meta_path = self.root / "manifest.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(exist_ok=True)
        self.blocks_dir.mkdir(exist_ok=True)
        self.index_dir.mkdir(exist_ok=True)
        if not self.meta_path.exists():
            dump_json(self.meta_path, {"documents": {}})

    def save_document(self, document: Document) -> None:
        dump_json(self.docs_dir / f"{document.doc_id}.json", document.to_dict())
        manifest = self._load_manifest()
        manifest["documents"][document.doc_id] = {
            "doc_id": document.doc_id,
            "title": document.title,
            "source_path": document.source_path,
            "page_count": document.page_count,
        }
        dump_json(self.meta_path, manifest)

    def load_document(self, doc_id: str) -> Document:
        return Document.from_dict(load_json(self.docs_dir / f"{doc_id}.json"))

    def list_documents(self) -> list[Document]:
        manifest = self._load_manifest()
        return [self.load_document(doc_id) for doc_id in sorted(manifest["documents"])]

    def save_blocks(self, doc_id: str, blocks: list[Block]) -> None:
        payload = {"blocks": [block.to_dict() for block in blocks]}
        dump_json(self.blocks_dir / f"{doc_id}.json", payload)

    def load_blocks(self, doc_id: str) -> list[Block]:
        payload = load_json(self.blocks_dir / f"{doc_id}.json")
        return [Block.from_dict(item) for item in payload.get("blocks", [])]

    def load_block(self, block_id: str) -> Block | None:
        doc_id = block_id.split(":", 1)[0]
        for block in self.load_blocks(doc_id):
            if block.block_id == block_id:
                return block
        return None

    def all_blocks(self, doc_ids: list[str] | None = None) -> list[Block]:
        documents = doc_ids or [document.doc_id for document in self.list_documents()]
        blocks: list[Block] = []
        for doc_id in documents:
            blocks.extend(self.load_blocks(doc_id))
        return blocks

    def save_index_metadata(self, payload: dict[str, Any]) -> None:
        dump_json(self.index_dir / "index_manifest.json", payload)

    def load_index_metadata(self) -> dict[str, Any]:
        path = self.index_dir / "index_manifest.json"
        if not path.exists():
            return {}
        return load_json(path)

    def save_payload(self, relative_path: str, payload: dict[str, Any]) -> None:
        dump_json(self.index_dir / relative_path, payload)

    def load_payload(self, relative_path: str) -> dict[str, Any]:
        return load_json(self.index_dir / relative_path)

    def _load_manifest(self) -> dict[str, Any]:
        return load_json(self.meta_path)
