from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-/.]*")
PROVIDER_SAFE_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-/.]*|[^\s]")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def truncate_text_by_token_count(text: str, max_tokens: int) -> str:
    """
    Conservatively truncate text to a provider-safe token budget.

    This counts words and also standalone punctuation/symbol characters so it
    overestimates token usage rather than underestimates it. That keeps the
    library on the safe side of provider-enforced input limits.
    """
    if not text or max_tokens <= 0:
        return ""

    matches = list(PROVIDER_SAFE_TOKEN_RE.finditer(text))
    if len(matches) <= max_tokens:
        return text

    cutoff = matches[max_tokens - 1].end()
    truncated = text[:cutoff].rstrip()
    suffix = "\n\n[Truncated for token limit]"
    return truncated + suffix


def estimate_provider_safe_token_count(text: str) -> int:
    return len(PROVIDER_SAFE_TOKEN_RE.findall(text or ""))


def truncate_text_by_char_budget(text: str, max_chars: int) -> str:
    """
    Hard character-budget truncation used as an additional provider-safe guardrail.
    """
    if not text or max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[Truncated for char limit]"


def unique_terms(tokens: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            result.append(token)
    return result


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    numerator = sum(x * y for x, y in zip(a, b))
    denom_a = math.sqrt(sum(x * x for x in a))
    denom_b = math.sqrt(sum(y * y for y in b))
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return numerator / (denom_a * denom_b)


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def preview_text(text: str, limit: int = 220) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def sentence_split(text: str) -> list[str]:
    stripped = " ".join((text or "").split())
    if not stripped:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", stripped) if part.strip()]


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def infer_title_from_path(path: str) -> str:
    return Path(path).stem.replace("_", " ").replace("-", " ").strip().title()


def format_citation(title: str, page_start: int, page_end: int, block_id: str) -> str:
    if page_start == page_end:
        page_text = f"p.{page_start}"
    else:
        page_text = f"pp.{page_start}-{page_end}"
    return f"{title} ({page_text}, {block_id})"
