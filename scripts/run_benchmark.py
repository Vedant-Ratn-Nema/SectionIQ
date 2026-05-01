#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sectioniq import SectionIQ
from sectioniq.benchmarks import BenchmarkHarness


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval benchmarks for SectionIQ.")
    parser.add_argument("--store", default=".sectioniq", help="Path to the local store.")
    parser.add_argument("--dataset", required=True, help="JSONL benchmark dataset.")
    parser.add_argument("--top-k", type=int, default=5, help="Cutoff for recall@k.")
    args = parser.parse_args()

    engine = SectionIQ(store_path=args.store)
    harness = BenchmarkHarness(engine)
    metrics = harness.evaluate(args.dataset, top_k=args.top_k)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
