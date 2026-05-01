#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "public_corpus_manifest.json"


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def download_file(url: str, target: Path, timeout: int = 120) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=timeout) as response:
        target.write_bytes(response.read())


def pdf_page_count(path: Path) -> int:
    with path.open("rb") as handle:
        return len(PdfReader(handle).pages)


def prepare_public_corpus(manifest_path: Path, force: bool = False) -> list[dict]:
    manifest = load_manifest(manifest_path)
    local_dir = ROOT / manifest["local_dir"]
    results = []
    for source in manifest["sources"]:
        target = local_dir / source["filename"]
        if force or not target.exists():
            print(f"Downloading {source['filename']} ...", flush=True)
            download_file(source["download_url"], target)
        actual_pages = pdf_page_count(target)
        expected_pages = int(source["page_count"])
        status = "ok" if actual_pages == expected_pages else "page_count_mismatch"
        results.append(
            {
                "id": source["id"],
                "filename": str(target),
                "expected_pages": expected_pages,
                "actual_pages": actual_pages,
                "status": status,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the public SectionIQ benchmark corpus.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to the public corpus manifest.")
    parser.add_argument("--force", action="store_true", help="Redownload PDFs even when local files exist.")
    args = parser.parse_args()

    results = prepare_public_corpus(Path(args.manifest), force=args.force)
    print(json.dumps({"results": results}, indent=2))
    failures = [item for item in results if item["status"] != "ok"]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
