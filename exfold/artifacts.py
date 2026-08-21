"""Calibration artifact validation shared by setup and tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


EXPECTED_SHA256 = {
    "qwen3-30b-a3b.pt": "3b64e8797c9cb2e4aa1329341229df0242b365f7e45a361f0fff3bc481325017",
    "deepseek-v4-flash.pt": "8cb5fb81f4439da72d2cfb72d5cbf180107a8c0bd9895e18bfc46a1063108d7f",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path) -> None:
    expected = EXPECTED_SHA256.get(path.name)
    if expected is not None and sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch for {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if path.name == "qwen3-30b-a3b.pt":
        shape = (48, 128, 128)
        if tuple(payload["coeff"].shape) != shape or tuple(payload["loss"].shape) != shape:
            raise ValueError("Qwen3 matrix shape mismatch")
        if tuple(payload["expert_norms"].shape) != shape[:2]:
            raise ValueError("Qwen3 norm shape mismatch")
    elif path.name == "deepseek-v4-flash.pt":
        shape = (43, 256, 256)
        keys = (
            "prefill_coeff",
            "prefill_selection_loss",
            "prefill_confidence_loss",
            "decode_coeff",
            "decode_loss",
        )
        if any(tuple(payload[key].shape) != shape for key in keys):
            raise ValueError("DeepSeek-V4 matrix shape mismatch")
        if tuple(payload["norms"].shape) != shape[:2]:
            raise ValueError("DeepSeek-V4 norm shape mismatch")
    else:
        raise ValueError(f"Unknown artifact: {path.name}")
