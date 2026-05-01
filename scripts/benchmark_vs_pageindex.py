#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import uuid

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
VENDOR = ROOT / ".vendor"
PAGEINDEX_ROOT = Path("/tmp/PageIndex")

for path in [str(VENDOR), str(SRC), str(PAGEINDEX_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from dotenv import load_dotenv

from openai import OpenAI
import PyPDF2

from pageindex import PageIndexClient

page_index_module = importlib.import_module("pageindex.page_index")


DEFAULT_PDF = ""
DEFAULT_CSV = ""


@dataclass
class CaseRow:
    row_index: int
    case_number: str
    subject: str
    description: str
    primary_part: str
    manual_answerability: str
    top_block_type: str
    top_citation: str
    sectioniq_answer: str
    historical_answer: str


def load_cases(csv_path: str, limit: int | None = None) -> list[CaseRow]:
    rows: list[CaseRow] = []
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            rows.append(
                CaseRow(
                    row_index=idx,
                    case_number=(row.get("case_number") or "").strip(),
                    subject=(row.get("subject") or "").strip(),
                    description=(row.get("description") or "").strip(),
                    primary_part=(row.get("primary_part") or "").strip(),
                    manual_answerability=(row.get("manual_answerability") or "").strip(),
                    top_block_type=(row.get("top_block_type") or "").strip(),
                    top_citation=(row.get("top_citation") or "").strip(),
                    sectioniq_answer=(row.get("llm_answer") or "").strip(),
                    historical_answer=(row.get("historical_answer") or "").strip(),
                )
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def make_query(case: CaseRow) -> str:
    parts = []
    if case.subject:
        parts.append(case.subject)
    if case.primary_part:
        parts.append(f"Primary part: {case.primary_part}")
    if case.description:
        parts.append(case.description)
    return "\n".join(parts).strip()


def get_client(model: str, workspace: str) -> PageIndexClient:
    return PageIndexClient(model=model, retrieve_model=model, workspace=workspace)


@contextmanager
def force_pageindex_no_toc_mode():
    original = page_index_module.check_toc
    page_index_module.check_toc = lambda page_list, opt=None: {
        "toc_content": None,
        "toc_page_list": [],
        "page_index_given_in_toc": "no",
    }
    try:
        yield
    finally:
        page_index_module.check_toc = original


@contextmanager
def pageindex_strict_toc_heading_detector():
    original = page_index_module.toc_detector_single_page

    def patched(content: str, model: str | None = None) -> str:
        head = (content or "")[:1500].upper().replace("\n", " ")
        if "TABLE OF CONTENTS" in head or "TABLE OF CONTEN TS" in head:
            return "yes"
        return "no"

    page_index_module.toc_detector_single_page = patched
    try:
        yield
    finally:
        page_index_module.toc_detector_single_page = original


@contextmanager
def pageindex_smaller_groups(max_tokens: int = 8000, overlap_page: int = 1):
    original = page_index_module.page_list_to_group_text
    page_index_module.page_list_to_group_text = (
        lambda page_contents, token_lengths, max_tokens=max_tokens, overlap_page=overlap_page: original(
            page_contents,
            token_lengths,
            max_tokens=max_tokens,
            overlap_page=overlap_page,
        )
    )
    try:
        yield
    finally:
        page_index_module.page_list_to_group_text = original


def _persist_custom_index(client: PageIndexClient, file_path: str, result: dict[str, Any]) -> str:
    doc_id = str(uuid.uuid4())
    pages = []
    with open(file_path, "rb") as handle:
        pdf_reader = PyPDF2.PdfReader(handle)
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            pages.append({"page": page_num, "content": page.extract_text() or ""})
    client.documents[doc_id] = {
        "id": doc_id,
        "type": "pdf",
        "path": file_path,
        "doc_name": result.get("doc_name", ""),
        "doc_description": result.get("doc_description", ""),
        "page_count": len(pages),
        "structure": result["structure"],
        "pages": pages,
    }
    if client.workspace:
        client._save_doc(doc_id)
    return doc_id


def index_pdf_with_pageindex_fallback(client: PageIndexClient, pdf_path: str) -> tuple[str, str]:
    with force_pageindex_no_toc_mode(), pageindex_smaller_groups():
        result = page_index_module.page_index(
            doc=pdf_path,
            model=client.model,
            max_page_num_each_node=100000,
            max_token_num_each_node=100000000,
            if_add_node_summary="yes",
            if_add_node_text="no",
            if_add_node_id="yes",
            if_add_doc_description="no",
        )
    return _persist_custom_index(client, pdf_path, result), "fallback_no_toc_custom"


def index_pdf_with_pageindex_toc_hint(client: PageIndexClient, pdf_path: str) -> tuple[str, str]:
    with pageindex_strict_toc_heading_detector():
        result = page_index_module.page_index(
            doc=pdf_path,
            model=client.model,
            toc_check_page_num=40,
            max_page_num_each_node=100000,
            max_token_num_each_node=100000000,
            if_add_node_summary="yes",
            if_add_node_text="no",
            if_add_node_id="yes",
            if_add_doc_description="no",
        )
    return _persist_custom_index(client, pdf_path, result), "toc_heading_hint"


def ensure_indexed(client: PageIndexClient, pdf_path: str, reindex: bool = False, force_no_toc: bool = False) -> tuple[str, str]:
    pdf_path = str(Path(pdf_path).expanduser().resolve())
    if not reindex:
        for doc_id, doc in client.documents.items():
            if doc.get("path") == pdf_path:
                return doc_id, "cached"
    if force_no_toc:
        return index_pdf_with_pageindex_fallback(client, pdf_path)
    try:
        return client.index(pdf_path), "default"
    except Exception as exc:
        if "toc transformation" not in str(exc).lower():
            raise
        print("PageIndex TOC transformation failed; retrying with stricter TOC heading detection.", flush=True)
        try:
            return index_pdf_with_pageindex_toc_hint(client, pdf_path)
        except Exception:
            print("TOC-heading retry failed; retrying with no-TOC fallback.", flush=True)
            return index_pdf_with_pageindex_fallback(client, pdf_path)


def flatten_tree(nodes: list[dict[str, Any]], parent_path: list[str] | None = None, depth: int = 0) -> list[dict[str, Any]]:
    parent_path = parent_path or []
    flat: list[dict[str, Any]] = []
    for node in nodes:
        title = str(node.get("title", "")).strip()
        path = parent_path + ([title] if title else [])
        flat.append(
            {
                "node_id": node.get("node_id"),
                "title": title,
                "path": path,
                "start_index": node.get("start_index"),
                "end_index": node.get("end_index"),
                "summary": str(node.get("summary", "")).strip(),
                "depth": depth,
                "children": node.get("nodes") or [],
            }
        )
        children = node.get("nodes") or []
        if children:
            flat.extend(flatten_tree(children, parent_path=path, depth=depth + 1))
    return flat


def score_node_overlap(query: str, node: dict[str, Any]) -> float:
    query_tokens = {token.lower() for token in query.split() if token}
    node_text = " ".join(
        [
            node.get("title", ""),
            " ".join(node.get("path", [])),
            node.get("summary", ""),
        ]
    ).lower()
    if not query_tokens:
        return 0.0
    hits = sum(1 for token in query_tokens if token in node_text)
    return hits / len(query_tokens)


class JsonLLM:
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def complete_json(self, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        content = response.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(content[start : end + 1])
            raise


def select_nodes_for_query(
    llm: JsonLLM,
    query: str,
    structure: list[dict[str, Any]],
    beam_width: int = 2,
    max_depth: int = 4,
) -> dict[str, Any]:
    node_lookup = {node["node_id"]: node for node in flatten_tree(structure)}
    frontier = [
        {
            "node_id": node.get("node_id"),
            "title": node.get("title", ""),
            "path": [node.get("title", "")] if node.get("title") else [],
            "start_index": node.get("start_index"),
            "end_index": node.get("end_index"),
            "summary": str(node.get("summary", "")).strip(),
            "depth": 0,
            "children": node.get("nodes") or [],
        }
        for node in structure
    ]
    selected_trace: list[dict[str, Any]] = []
    terminal_nodes: list[dict[str, Any]] = []

    for depth in range(max_depth):
        if not frontier:
            break
        candidates = []
        for node in frontier:
            candidates.append(
                {
                    "node_id": node["node_id"],
                    "title": node["title"],
                    "path": " > ".join(part for part in node["path"] if part),
                    "page_range": f"{node['start_index']}-{node['end_index']}",
                    "summary": node["summary"][:700],
                }
            )
        prompt = {
            "query": query,
            "depth": depth,
            "beam_width": beam_width,
            "candidates": candidates,
            "instruction": (
                "Pick the most promising node_ids for answering the query. Prefer the most specific relevant sections. "
                "If nothing looks relevant, return an empty list."
            ),
            "output_schema": {
                "selected_node_ids": ["string"],
                "reason": "string",
            },
        }
        parsed = llm.complete_json(
            system_prompt=(
                "You are selecting relevant sections from a PageIndex tree for document question answering. "
                "Return strict JSON only."
            ),
            payload=prompt,
        )
        selected_ids = [
            str(node_id)
            for node_id in parsed.get("selected_node_ids", [])
            if isinstance(node_id, str) and node_id in {node["node_id"] for node in frontier}
        ]
        if not selected_ids:
            ranked = sorted(frontier, key=lambda item: score_node_overlap(query, item), reverse=True)
            selected_ids = [item["node_id"] for item in ranked[:beam_width] if score_node_overlap(query, item) > 0]
        selected_nodes = [node for node in frontier if node["node_id"] in set(selected_ids)]
        if not selected_nodes:
            break
        selected_trace.extend(
            {
                "depth": depth,
                "node_id": node["node_id"],
                "title": node["title"],
                "path": " > ".join(part for part in node["path"] if part),
                "start_index": node["start_index"],
                "end_index": node["end_index"],
            }
            for node in selected_nodes
        )
        next_frontier = []
        for node in selected_nodes:
            children = node.get("children") or []
            if children:
                for child in children:
                    child_title = str(child.get("title", "")).strip()
                    next_frontier.append(
                        {
                            "node_id": child.get("node_id"),
                            "title": child_title,
                            "path": node["path"] + ([child_title] if child_title else []),
                            "start_index": child.get("start_index"),
                            "end_index": child.get("end_index"),
                            "summary": str(child.get("summary", "")).strip(),
                            "depth": depth + 1,
                            "children": child.get("nodes") or [],
                        }
                    )
            else:
                terminal_nodes.append(node)
        if not next_frontier:
            terminal_nodes.extend([node for node in selected_nodes if node not in terminal_nodes])
            break
        frontier = next_frontier

    if not terminal_nodes and selected_trace:
        for item in reversed(selected_trace[-beam_width:]):
            node = node_lookup.get(item["node_id"])
            if node:
                terminal_nodes.append(node)
    if not terminal_nodes:
        ranked = sorted(flatten_tree(structure), key=lambda item: score_node_overlap(query, item), reverse=True)
        terminal_nodes = ranked[:beam_width]

    return {
        "selected_trace": selected_trace,
        "terminal_nodes": terminal_nodes[:beam_width],
    }


def pages_from_nodes(nodes: list[dict[str, Any]], max_total_pages: int = 12) -> list[int]:
    pages: list[int] = []
    for node in nodes:
        start = int(node.get("start_index") or 0)
        end = int(node.get("end_index") or start)
        if start <= 0:
            continue
        for page in range(start, end + 1):
            if page not in pages:
                pages.append(page)
            if len(pages) >= max_total_pages:
                return pages
    return pages


def compress_pages(pages: list[int]) -> str:
    if not pages:
        return ""
    ordered = sorted(set(pages))
    ranges: list[str] = []
    start = prev = ordered[0]
    for page in ordered[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = page
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def answer_with_pageindex(
    llm: JsonLLM,
    client: PageIndexClient,
    doc_id: str,
    query: str,
    structure: list[dict[str, Any]],
) -> dict[str, Any]:
    selection = select_nodes_for_query(llm, query, structure)
    pages = pages_from_nodes(selection["terminal_nodes"])
    page_spec = compress_pages(pages)
    page_content = []
    if page_spec:
        page_content = json.loads(client.get_page_content(doc_id, page_spec))
    parsed = llm.complete_json(
        system_prompt=(
            "You answer questions from PageIndex-retrieved page content. "
            "Use only the supplied evidence. If the evidence is insufficient, say so plainly. "
            "Return strict JSON only."
        ),
        payload={
            "query": query,
            "selected_nodes": [
                {
                    "node_id": node.get("node_id"),
                    "title": node.get("title"),
                    "path": node.get("path"),
                    "start_index": node.get("start_index"),
                    "end_index": node.get("end_index"),
                    "summary": node.get("summary", ""),
                }
                for node in selection["terminal_nodes"]
            ],
            "page_content": page_content,
            "output_schema": {
                "answer": "string",
                "confidence": "number between 0 and 1",
            },
        },
    )
    return {
        "answer": str(parsed.get("answer") or "").strip(),
        "confidence": parsed.get("confidence"),
        "page_spec": page_spec,
        "page_content_pages": [item.get("page") for item in page_content if isinstance(item, dict)],
        "selected_trace": selection["selected_trace"],
        "selected_nodes": [
            {
                "node_id": node.get("node_id"),
                "title": node.get("title"),
                "path": node.get("path"),
                "start_index": node.get("start_index"),
                "end_index": node.get("end_index"),
            }
            for node in selection["terminal_nodes"]
        ],
    }


def judge_answers(
    llm: JsonLLM,
    case: CaseRow,
    query: str,
    pageindex_answer: str,
) -> dict[str, Any]:
    parsed = llm.complete_json(
        system_prompt=(
            "You are an impartial evaluator comparing two answers to the same support query. "
            "Use the historical answer as the reference standard when available. "
            "Reward correctness and useful specificity over style. Return strict JSON only."
        ),
        payload={
            "query": query,
            "historical_answer": case.historical_answer,
            "candidate_a_name": "sectioniq",
            "candidate_a_answer": case.sectioniq_answer,
            "candidate_b_name": "pageindex",
            "candidate_b_answer": pageindex_answer,
            "output_schema": {
                "winner": "sectioniq | pageindex | tie",
                "sectioniq_score": "integer 1-5",
                "pageindex_score": "integer 1-5",
                "reason": "string",
            },
        },
    )
    return {
        "winner": parsed.get("winner", "tie"),
        "sectioniq_score": parsed.get("sectioniq_score"),
        "pageindex_score": parsed.get("pageindex_score"),
        "reason": parsed.get("reason", ""),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "evaluated_cases": len(results),
        "sectioniq_wins": 0,
        "pageindex_wins": 0,
        "ties": 0,
        "sectioniq_score_total": 0,
        "pageindex_score_total": 0,
        "by_manual_answerability": {},
    }
    for item in results:
        judge = item.get("judge", {})
        winner = judge.get("winner", "tie")
        if winner == "sectioniq":
            summary["sectioniq_wins"] += 1
        elif winner == "pageindex":
            summary["pageindex_wins"] += 1
        else:
            summary["ties"] += 1
        try:
            summary["sectioniq_score_total"] += int(judge.get("sectioniq_score") or 0)
            summary["pageindex_score_total"] += int(judge.get("pageindex_score") or 0)
        except (TypeError, ValueError):
            pass
        bucket = item["case"]["manual_answerability"]
        slot = summary["by_manual_answerability"].setdefault(
            bucket,
            {
                "count": 0,
                "sectioniq_wins": 0,
                "pageindex_wins": 0,
                "ties": 0,
            },
        )
        slot["count"] += 1
        if winner == "sectioniq":
            slot["sectioniq_wins"] += 1
        elif winner == "pageindex":
            slot["pageindex_wins"] += 1
        else:
            slot["ties"] += 1
    if results:
        summary["sectioniq_avg_score"] = round(
            summary["sectioniq_score_total"] / len(results), 3
        )
        summary["pageindex_avg_score"] = round(
            summary["pageindex_score_total"] / len(results), 3
        )
    else:
        summary["sectioniq_avg_score"] = 0.0
        summary["pageindex_avg_score"] = 0.0
    return summary


def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload.get("results", [])
    return payload


def save_checkpoint(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SectionIQ outputs against PageIndex.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="CSV with first-100 outputs.")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="PDF used for retrieval.")
    parser.add_argument("--workspace", default=str(ROOT / ".pageindex_workspace"), help="PageIndex workspace path.")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "pageindex_benchmark_results.json"), help="Detailed output JSON.")
    parser.add_argument(
        "--model",
        default=os.getenv("SECTIONIQ_LLM_MODEL", os.getenv("STRUCTURED_PDF_RAG_LLM_MODEL", "gpt-4o-mini")),
        help="Model for PageIndex retrieval and judging.",
    )
    parser.add_argument("--limit", type=int, default=100, help="Number of rows to benchmark.")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Save every N completed cases.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file if present.")
    parser.add_argument("--reindex", action="store_true", help="Force PageIndex reindexing.")
    parser.add_argument("--force-no-toc", action="store_true", help="Force PageIndex indexing to skip TOC processing.")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set.")

    cases = load_cases(args.csv, limit=args.limit)
    output_path = Path(args.output)
    results = load_checkpoint(output_path) if args.resume else []
    completed = {item["case"]["case_number"] for item in results}

    client = get_client(model=args.model, workspace=args.workspace)
    doc_id, index_mode = ensure_indexed(client, args.pdf, reindex=args.reindex, force_no_toc=args.force_no_toc)
    print(f"PageIndex doc_id: {doc_id} (mode={index_mode})", flush=True)
    structure = json.loads(client.get_document_structure(doc_id))
    llm = JsonLLM(model=args.model)

    total = len(cases)
    for idx, case in enumerate(cases, start=1):
        if case.case_number in completed:
            continue
        query = make_query(case)
        started = time.time()
        try:
            pageindex_result = answer_with_pageindex(llm, client, doc_id, query, structure)
            judge = judge_answers(llm, case, query, pageindex_result["answer"])
            item = {
                "case": {
                    "case_number": case.case_number,
                    "row_index": case.row_index,
                    "subject": case.subject,
                    "primary_part": case.primary_part,
                    "manual_answerability": case.manual_answerability,
                    "historical_answer": case.historical_answer,
                },
                "query": query,
                "pageindex_index_mode": index_mode,
                "sectioniq_answer": case.sectioniq_answer,
                "pageindex": pageindex_result,
                "judge": judge,
                "elapsed_seconds": round(time.time() - started, 3),
            }
        except Exception as exc:
            item = {
                "case": {
                    "case_number": case.case_number,
                    "row_index": case.row_index,
                    "subject": case.subject,
                    "primary_part": case.primary_part,
                    "manual_answerability": case.manual_answerability,
                    "historical_answer": case.historical_answer,
                },
                "query": query,
                "pageindex_index_mode": index_mode,
                "sectioniq_answer": case.sectioniq_answer,
                "error": str(exc),
                "elapsed_seconds": round(time.time() - started, 3),
            }
        results.append(item)
        completed.add(case.case_number)
        if len(results) % args.checkpoint_every == 0 or idx == total:
            save_checkpoint(output_path, results)
            print(f"Checkpoint saved: {output_path} ({len(results)}/{total})")

    summary = summarize_results(results)
    final_payload = {
        "summary": summary,
        "results": results,
    }
    save_checkpoint(output_path, final_payload["results"])
    output_path.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
