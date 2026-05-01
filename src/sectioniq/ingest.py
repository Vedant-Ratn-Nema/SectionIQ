from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .models import Block, Document, TableBlock
from .utils import infer_title_from_path, normalize_whitespace, stable_hash


NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,4})[\s\-:]+(.+)$")
LIST_ITEM_RE = re.compile(r"^(?:[-*•]|\d+[.)])\s+")
MULTISPACE_RE = re.compile(r"\s{2,}")
EXPLICIT_WARNING_RE = re.compile(r"^(?:warning|caution|danger)\s*[:.\-]\s+\S+", re.I)
EXPLICIT_NOTE_RE = re.compile(r"^(?:note|important)\s*[:.\-]\s+\S+", re.I)
CODE_TOKEN_RE = re.compile(r"^[A-Z]?\d{1,4}(?:[-.]\d{1,4}){1,5}[A-Z]?$", re.I)
PART_TOKEN_RE = re.compile(r"^(?=.*\d)[A-Z0-9]+(?:[-_/][A-Z0-9]+){1,6}$", re.I)
NUMBER_TOKEN_RE = re.compile(r"^\d+(?:[.,]\d+)?$")


@dataclass
class ParsedPDF:
    source_path: str
    title: str
    pages: list[str]
    metadata: dict[str, Any]


class PDFParser:
    def parse(
        self,
        path: str,
        max_pages: int | None = None,
        page_range: tuple[int, int] | None = None,
    ) -> ParsedPDF:
        reader = PdfReader(path)
        metadata = reader.metadata or {}
        title = getattr(metadata, "title", None) or metadata.get("/Title") or infer_title_from_path(path)
        pages = []
        start_idx = 0
        end_idx = len(reader.pages)
        if page_range is not None:
            start_idx = max(page_range[0] - 1, 0)
            end_idx = min(page_range[1], len(reader.pages))
        elif max_pages is not None:
            end_idx = min(max_pages, len(reader.pages))

        for page in reader.pages[start_idx:end_idx]:
            text = page.extract_text() or ""
            pages.append(text)
        return ParsedPDF(
            source_path=str(Path(path).expanduser().resolve()),
            title=normalize_whitespace(title) or infer_title_from_path(path),
            pages=pages,
            metadata={
                "pdf_metadata": {str(key): str(value) for key, value in metadata.items()} if hasattr(metadata, "items") else {}
            },
        )


class IngestionPipeline:
    def __init__(self, parser: PDFParser | None = None):
        self.parser = parser or PDFParser()

    def ingest_file(
        self,
        path: str,
        metadata: dict[str, Any] | None = None,
        max_pages: int | None = None,
        page_range: tuple[int, int] | None = None,
    ) -> tuple[Document, list[Block]]:
        parsed = self.parser.parse(path, max_pages=max_pages, page_range=page_range)
        return self.ingest_pages(
            source_path=parsed.source_path,
            title=parsed.title,
            pages=parsed.pages,
            metadata={
                **parsed.metadata,
                **(metadata or {}),
                "ingest_options": {
                    "max_pages": max_pages,
                    "page_range": list(page_range) if page_range else None,
                },
            },
        )

    def ingest_pages(
        self,
        source_path: str,
        title: str,
        pages: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Document, list[Block]]:
        doc_id = self._make_doc_id(source_path, pages)
        blocks: list[Block] = []
        current_path: list[dict[str, Any]] = []
        block_counter = 0
        heading_candidates = 0
        table_count = 0
        suspected_table_rows = 0
        warning_count = 0
        note_count = 0

        for page_num, page_text in enumerate(pages, start=1):
            lines = [line.rstrip() for line in page_text.splitlines()]
            i = 0
            while i < len(lines):
                raw_line = lines[i]
                line = normalize_whitespace(raw_line)
                if not line:
                    i += 1
                    continue

                if self._looks_like_table_row(raw_line):
                    raw_rows = [raw_line]
                    j = i + 1
                    while j < len(lines):
                        candidate_raw = lines[j]
                        candidate = normalize_whitespace(candidate_raw)
                        if not candidate or not self._looks_like_table_row(candidate_raw):
                            break
                        raw_rows.append(candidate_raw)
                        j += 1
                    block_counter += 1
                    table_count += 1
                    suspected_table_rows += len(raw_rows)
                    table = self._build_table_block(
                        doc_id=doc_id,
                        block_id=f"{doc_id}:b{block_counter:05d}",
                        page_num=page_num,
                        rows=raw_rows,
                        current_path=current_path,
                    )
                    blocks.append(table)
                    i = j
                    continue

                heading_info = self._detect_heading(line)
                if heading_info:
                    heading_candidates += 1
                    current_path = self._update_section_path(current_path, heading_info)
                    block_counter += 1
                    section_titles = [item["title"] for item in current_path]
                    parent_id = current_path[-2]["block_id"] if len(current_path) > 1 else None
                    block_id = f"{doc_id}:b{block_counter:05d}"
                    current_path[-1]["block_id"] = block_id
                    blocks.append(
                        Block(
                            block_id=block_id,
                            doc_id=doc_id,
                            page_start=page_num,
                            page_end=page_num,
                            block_type="section",
                            text=line,
                            parent_id=parent_id,
                            section_path=section_titles,
                            metadata={
                                "heading_level": heading_info["level"],
                                "neighboring_headings": {
                                    "current": section_titles[-1],
                                    "parent": section_titles[-2] if len(section_titles) > 1 else None,
                                },
                            },
                        )
                    )
                    i += 1
                    continue

                if LIST_ITEM_RE.match(line):
                    item_lines = [line]
                    j = i + 1
                    while j < len(lines):
                        candidate = normalize_whitespace(lines[j])
                        if not candidate or LIST_ITEM_RE.match(candidate) or self._detect_heading(candidate):
                            break
                        item_lines.append(candidate)
                        j += 1
                    block_counter += 1
                    block_type = self._classify_text_block(" ".join(item_lines))
                    if block_type == "warning":
                        warning_count += 1
                    elif block_type == "note":
                        note_count += 1
                    blocks.append(
                        self._make_text_block(
                            doc_id=doc_id,
                            block_id=f"{doc_id}:b{block_counter:05d}",
                            page_num=page_num,
                            block_type=block_type,
                            text=" ".join(item_lines),
                            current_path=current_path,
                        )
                    )
                    i = j
                    continue

                paragraph_lines = [line]
                j = i + 1
                while j < len(lines):
                    candidate_raw = lines[j]
                    candidate = normalize_whitespace(candidate_raw)
                    if not candidate:
                        if paragraph_lines:
                            break
                        j += 1
                        continue
                    if self._detect_heading(candidate) or self._looks_like_table_row(candidate_raw) or LIST_ITEM_RE.match(candidate):
                        break
                    paragraph_lines.append(candidate)
                    j += 1
                paragraph_text = " ".join(paragraph_lines)
                block_counter += 1
                block_type = self._classify_text_block(paragraph_text)
                if block_type == "warning":
                    warning_count += 1
                elif block_type == "note":
                    note_count += 1
                blocks.append(
                    self._make_text_block(
                        doc_id=doc_id,
                        block_id=f"{doc_id}:b{block_counter:05d}",
                        page_num=page_num,
                        block_type=block_type,
                        text=paragraph_text,
                        current_path=current_path,
                    )
                )
                i = j

        extraction_flags = {
            "weak_headings": heading_candidates == 0,
            "has_tables": table_count > 0,
            "table_count": table_count,
            "suspected_table_rows": suspected_table_rows,
            "warning_count": warning_count,
            "note_count": note_count,
            "heading_count": heading_candidates,
            "heading_density": round(heading_candidates / max(len(pages), 1), 3),
            "block_count": len(blocks),
            "page_text_coverage": round(sum(1 for page in pages if page.strip()) / max(len(pages), 1), 3),
        }
        document = Document(
            doc_id=doc_id,
            source_path=source_path,
            title=title,
            page_count=len(pages),
            metadata=metadata or {},
            extraction_flags=extraction_flags,
        )
        return document, blocks

    def _make_doc_id(self, source_path: str, pages: list[str]) -> str:
        seed = f"{Path(source_path).resolve()}::{len(pages)}::{stable_hash(''.join(pages)[:2000])}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

    def _detect_heading(self, line: str) -> dict[str, Any] | None:
        stripped = line.strip()
        if not stripped or len(stripped.split()) > 14 or len(stripped) > 120:
            return None
        if stripped.endswith(".") and len(stripped.split()) > 5:
            return None

        numbered = NUMBERED_HEADING_RE.match(stripped)
        if numbered:
            structure = numbered.group(1)
            title = normalize_whitespace(numbered.group(2))
            return {"title": title, "level": structure.count(".") + 1, "raw": stripped}

        if any(char.isdigit() for char in stripped):
            return None

        alpha_chars = [char for char in stripped if char.isalpha()]
        upper_ratio = (
            sum(1 for char in alpha_chars if char.isupper()) / len(alpha_chars)
            if alpha_chars
            else 0.0
        )
        title_case_words = [word for word in stripped.split() if word[:1].isupper()]
        has_colon = stripped.endswith(":")
        if upper_ratio > 0.75 and len(alpha_chars) >= 4:
            return {"title": stripped.rstrip(":"), "level": 1, "raw": stripped}
        if len(title_case_words) >= max(1, len(stripped.split()) - 1) and len(stripped.split()) <= 8:
            return {"title": stripped.rstrip(":"), "level": 2 if not has_colon else 3, "raw": stripped}
        return None

    def _update_section_path(self, current_path: list[dict[str, Any]], heading_info: dict[str, Any]) -> list[dict[str, Any]]:
        level = heading_info["level"]
        trimmed = [entry for entry in current_path if entry["level"] < level]
        trimmed.append({"title": heading_info["title"], "level": level, "block_id": None})
        return trimmed

    def _looks_like_table_row(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped or EXPLICIT_WARNING_RE.match(stripped) or EXPLICIT_NOTE_RE.match(stripped):
            return False
        if "|" in stripped and stripped.count("|") >= 1:
            return True
        columns = [part for part in MULTISPACE_RE.split(stripped) if part]
        if len(columns) >= 3:
            short_columns = sum(1 for part in columns if len(part.split()) <= 6)
            return short_columns >= 2
        return self._looks_like_collapsed_table_row(stripped)

    def _looks_like_collapsed_table_row(self, line: str) -> bool:
        tokens = line.split()
        if len(tokens) < 4:
            return False
        if line.endswith((".", "?", "!")):
            return False

        code_hits = sum(1 for token in tokens if CODE_TOKEN_RE.match(token))
        part_hits = sum(1 for token in tokens if PART_TOKEN_RE.match(token))
        numeric_hits = sum(1 for token in tokens if NUMBER_TOKEN_RE.match(token.strip("(),;:")))
        uppercase_code_hits = sum(
            1
            for token in tokens
            if any(char.isdigit() for char in token) and token.upper() == token and len(token) >= 3
        )

        starts_with_code = CODE_TOKEN_RE.match(tokens[0]) is not None
        has_terminal_value = NUMBER_TOKEN_RE.match(tokens[-1].strip("(),;:")) is not None
        has_identifier_columns = part_hits >= 1 or uppercase_code_hits >= 2
        if starts_with_code and has_identifier_columns:
            return True
        if starts_with_code and has_terminal_value and numeric_hits >= 2:
            return True
        if has_terminal_value and has_identifier_columns and len(tokens) <= 14:
            return True
        return False

    def _build_table_block(
        self,
        doc_id: str,
        block_id: str,
        page_num: int,
        rows: list[str],
        current_path: list[dict[str, Any]],
    ) -> TableBlock:
        parsed_rows: list[list[str]] = []
        cells: list[dict[str, Any]] = []
        for row_idx, raw in enumerate(rows):
            if "|" in raw:
                columns = [normalize_whitespace(part) for part in raw.split("|") if normalize_whitespace(part)]
            else:
                columns = [normalize_whitespace(part) for part in MULTISPACE_RE.split(raw) if normalize_whitespace(part)]
                if len(columns) == 1:
                    columns = self._split_collapsed_table_row(raw)
            parsed_rows.append(columns)
            for col_idx, cell in enumerate(columns):
                cells.append({"row": row_idx, "col": col_idx, "text": cell})
        text = "\n".join(" | ".join(row) for row in parsed_rows)
        section_titles = [item["title"] for item in current_path]
        parent_id = current_path[-1]["block_id"] if current_path else None
        return TableBlock(
            block_id=block_id,
            doc_id=doc_id,
            page_start=page_num,
            page_end=page_num,
            block_type="table",
            text=text,
            parent_id=parent_id,
            section_path=section_titles,
            metadata={
                "neighboring_headings": {
                    "current": section_titles[-1] if section_titles else None,
                    "parent": section_titles[-2] if len(section_titles) > 1 else None,
                }
            },
            rows=parsed_rows,
            cells=cells,
        )

    def _split_collapsed_table_row(self, row: str) -> list[str]:
        tokens = normalize_whitespace(row).split()
        if len(tokens) < 4:
            return [normalize_whitespace(row)]
        columns: list[str] = []
        start = 0
        end = len(tokens)
        if CODE_TOKEN_RE.match(tokens[0]):
            columns.append(tokens[0])
            start = 1
        if start < end and PART_TOKEN_RE.match(tokens[start]):
            columns.append(tokens[start])
            start += 1
        if start < end and NUMBER_TOKEN_RE.match(tokens[-1].strip("(),;:")):
            end -= 1
            middle = " ".join(tokens[start:end]).strip()
            if middle:
                columns.append(middle)
            columns.append(tokens[-1].strip("(),;:"))
        else:
            middle = " ".join(tokens[start:end]).strip()
            if middle:
                columns.append(middle)
        return columns or [normalize_whitespace(row)]

    def _make_text_block(
        self,
        doc_id: str,
        block_id: str,
        page_num: int,
        block_type: str,
        text: str,
        current_path: list[dict[str, Any]],
    ) -> Block:
        section_titles = [item["title"] for item in current_path]
        parent_id = current_path[-1]["block_id"] if current_path else None
        return Block(
            block_id=block_id,
            doc_id=doc_id,
            page_start=page_num,
            page_end=page_num,
            block_type=block_type,
            text=normalize_whitespace(text),
            parent_id=parent_id,
            section_path=section_titles,
            metadata={
                "neighboring_headings": {
                    "current": section_titles[-1] if section_titles else None,
                    "parent": section_titles[-2] if len(section_titles) > 1 else None,
                }
            },
        )

    def _classify_text_block(self, text: str) -> str:
        lowered = text.lower()
        if EXPLICIT_WARNING_RE.match(text):
            return "warning"
        if EXPLICIT_NOTE_RE.match(text):
            return "note"
        if LIST_ITEM_RE.match(text):
            return "list_item"
        if re.match(r"^(figure|fig\.|image)\s+\d+", lowered):
            return "figure_caption"
        return "paragraph"
