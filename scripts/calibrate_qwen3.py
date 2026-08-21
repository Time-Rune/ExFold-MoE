#!/usr/bin/env python3
"""Calibrate ExFold pairwise scalar projectors on unlabeled text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

from exfold.qwen3.model_patch import attach_stats, prepare_qwen3_moe


def load_text_file(path: Path) -> list[str]:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path, columns=["text"])
        values = frame["text"].dropna().tolist()
    elif path.suffix == ".jsonl":
        values = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                values.append(item["text"] if isinstance(item, dict) else item)
    elif path.suffix == ".txt":
        values = path.read_text(encoding="utf-8").splitlines()
    else:
        raise ValueError(f"unsupported calibration file: {path}")
    return [str(value).strip() for value in values if str(value).strip()]


def load_texts(path: Path, num_samples: int) -> list[str]:
    suffixes = {".parquet", ".jsonl", ".txt"}
    files = (
        sorted(item for item in path.iterdir() if item.suffix in suffixes)
        if path.is_dir()
        else [path]
    )
    if not files:
        raise FileNotFoundError(f"no calibration files found under {path}")
    texts = [text for file_path in files for text in load_text_file(file_path)]
    if len(texts) < num_samples:
        raise ValueError(f"requested {num_samples} samples, but only found {len(texts)}")
    return texts[:num_samples]


@torch.no_grad()
def calibrate(args: argparse.Namespace) -> None:
    texts = load_texts(args.data, args.num_samples)
    model_class, moe_class = prepare_qwen3_moe()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = model_class.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    stats = attach_stats(
        model,
        moe_class,
        ridge=args.ridge,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )

    input_device = model.get_input_embeddings().weight.device
    for index, text in enumerate(texts, start=1):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        inputs = {name: value.to(input_device) for name, value in inputs.items()}
        model.model(**inputs, use_cache=False)
        print(f"[{index:02d}/{len(texts):02d}] tokens={inputs['input_ids'].shape[1]}")

    finalized = [item.finalize() for item in stats]
    coefficient = torch.stack([item[0].cpu() for item in finalized])
    projection_loss = torch.stack([item[1].cpu() for item in finalized])
    expert_norms = torch.stack([item.expert_norms().cpu() for item in stats])

    payload = {
        "format_version": 1,
        "coeff": coefficient,
        "loss": projection_loss,
        "expert_norms": expert_norms,
        "loss_kind": "relative_scalar_reconstruction_error",
        "weighting": "source_output_norm",
        "ridge": args.ridge,
        "clip_min": args.clip_min,
        "clip_max": args.clip_max,
        "num_samples": len(texts),
        "max_length": args.max_length,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"saved={args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qwen3-30b-a3b.pt"),
    )
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--clip-min", type=float, default=-4.0)
    parser.add_argument("--clip-max", type=float, default=4.0)
    args = parser.parse_args()
    if args.num_samples <= 0 or args.max_length <= 0:
        parser.error("--num-samples and --max-length must be positive")
    return args


if __name__ == "__main__":
    calibrate(parse_args())
