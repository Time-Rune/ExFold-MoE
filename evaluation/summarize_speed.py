#!/usr/bin/env python3
"""Pair baseline and ExFold vLLM benchmark JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("failed", 0) or payload.get("completed", 0) <= 0:
        raise ValueError(f"Incomplete benchmark: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--ttft-stat", choices=("mean", "median"), default="median"
    )
    args = parser.parse_args()
    rows = []
    for baseline_path in sorted((args.root / "original").glob("*.json")):
        suffix = baseline_path.name.removeprefix("original_")
        method_path = args.root / "exfold" / f"exfold_{suffix}"
        if not method_path.is_file():
            continue
        baseline, method = load(baseline_path), load(method_path)
        phase = "prefill" if "prefill" in suffix else "decode"
        metric = (
            f"{args.ttft_stat}_ttft_ms"
            if phase == "prefill"
            else "mean_tpot_ms"
        )
        base_value = float(baseline[metric])
        method_value = float(method[metric])
        rows.append(
            {
                "point": suffix.removesuffix(".json"),
                "metric": metric,
                "original_ms": base_value,
                "exfold_ms": method_value,
                "speedup": base_value / method_value,
                "original_output_tok_s": float(baseline["output_throughput"]),
                "exfold_output_tok_s": float(method["output_throughput"]),
            }
        )
    if not rows:
        raise SystemExit(f"No paired results under {args.root}")
    output = args.root / "summary.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['point']}: {row['original_ms']:.3f} -> "
            f"{row['exfold_ms']:.3f} ms ({row['speedup']:.3f}x)"
        )
    print(output)
