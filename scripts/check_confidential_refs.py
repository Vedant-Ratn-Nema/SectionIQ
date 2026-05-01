#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PATTERN_FILE = ROOT / ".confidential_patterns"

DEFAULT_PATTERNS: list[str] = []


def tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return [ROOT / line.strip() for line in output.splitlines() if line.strip()]


def scan_files(patterns: list[str]) -> list[tuple[Path, str]]:
    findings: list[tuple[Path, str]] = []
    for path in tracked_files():
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in patterns:
            if pattern in text:
                findings.append((path.relative_to(ROOT), pattern))
    return findings


def load_local_patterns() -> list[str]:
    patterns: list[str] = []
    env_patterns = os.getenv("SECTIONIQ_CONFIDENTIAL_PATTERNS", "")
    patterns.extend(item.strip() for item in env_patterns.splitlines() if item.strip())
    if LOCAL_PATTERN_FILE.exists():
        patterns.extend(item.strip() for item in LOCAL_PATTERN_FILE.read_text(encoding="utf-8").splitlines() if item.strip())
    return patterns


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail if tracked files contain known confidential release-blocking strings.")
    parser.add_argument("--pattern", action="append", default=[], help="Additional literal pattern to scan for.")
    args = parser.parse_args()

    findings = scan_files(DEFAULT_PATTERNS + load_local_patterns() + args.pattern)
    if findings:
        print("Confidential reference scan failed:")
        for path, pattern in findings:
            print(f"- {path}: {pattern}")
        raise SystemExit(1)
    print("Confidential reference scan passed.")


if __name__ == "__main__":
    main()
