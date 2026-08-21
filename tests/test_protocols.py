#!/usr/bin/env python3
"""Validate every generated OpenCompass config without loading datasets."""

from __future__ import annotations

import argparse
import py_compile
import tempfile
from pathlib import Path

from evaluation.opencompass_eval import apply_defaults, parse_tasks, write_config


def build_args(protocol: str, root: Path) -> argparse.Namespace:
    args = argparse.Namespace(
        protocol=protocol,
        model_name=None,
        model_path=Path("/models/placeholder"),
        api_base="http://127.0.0.1:8000/v1",
        work_dir=root,
        batch_size=None,
        max_workers=None,
        infer_workers=None,
        eval_workers=8,
        qps=None,
        seed=None,
        retry=3,
        timeout=21600,
        aime_repeats=None,
        code_repeats=None,
        limit=None,
        test_range=None,
    )
    apply_defaults(args)
    return args


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        counts = {"qwen3": 8, "deepseek-v4": 9}
        for protocol, expected in counts.items():
            args = build_args(protocol, root / protocol)
            tasks = parse_tasks(protocol, "paper")
            assert len(tasks) == expected
            for name in tasks:
                output = root / protocol / name / "config.py"
                output.parent.mkdir(parents=True, exist_ok=True)
                write_config(args, name, output)
                py_compile.compile(str(output), doraise=True)
    print("PASS")


if __name__ == "__main__":
    main()
