"""Calibration artifact validation shared by setup and tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


EXPECTED_SHA256 = {
    "qwen3-30b-a3b.pt": "3b64e8797c9cb2e4aa1329341229df0242b365f7e45a361f0fff3bc481325017",
    "deepseek-v4-flash.pt": "65198c7c291d37b62d85261117051ce3ac4419385c5927b3d1342d8a3fca9470",
}

_LOSS_SENTINEL = 1.0e20
_MAX_LAYER_MEDIAN_BEST_LOSS = 0.9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loss_health(name: str, matrix: torch.Tensor) -> dict[str, float]:
    loss = torch.as_tensor(matrix, dtype=torch.float32)
    if loss.ndim != 3 or loss.shape[-1] != loss.shape[-2]:
        raise ValueError(f"{name} must be a square [layers, experts, experts] matrix")

    experts = loss.shape[-1]
    available = torch.isfinite(loss) & (loss.abs() < _LOSS_SENTINEL)
    diagonal = torch.eye(
        experts, dtype=torch.bool, device=loss.device
    ).unsqueeze(0)
    available &= ~diagonal
    if torch.any(loss[available] < -1.0e-6):
        raise ValueError(f"{name} contains negative reconstruction losses")

    row_best = torch.where(available, loss, torch.inf).amin(dim=-1)
    layer_medians = []
    eligible_rows = 0
    for layer_best in row_best:
        eligible = layer_best[torch.isfinite(layer_best)]
        if eligible.numel() == 0:
            continue
        eligible_rows += int(eligible.numel())
        layer_medians.append(float(eligible.median()))
    if not layer_medians:
        raise ValueError(f"{name} has no available off-diagonal expert pairs")

    worst_layer_median = max(layer_medians)
    if worst_layer_median >= _MAX_LAYER_MEDIAN_BEST_LOSS:
        raise ValueError(
            f"{name} is degenerate: worst per-layer median best-pair loss "
            f"is {worst_layer_median:.6f} (expected < "
            f"{_MAX_LAYER_MEDIAN_BEST_LOSS})"
        )
    return {
        "eligible_rows": float(eligible_rows),
        "median_best_loss": float(row_best[torch.isfinite(row_best)].median()),
        "worst_layer_median_best_loss": worst_layer_median,
    }


def validate_deepseek_payload(
    payload: dict, shape: tuple[int, int, int] = (43, 256, 256)
) -> dict[str, dict[str, float]]:
    if not isinstance(payload, dict):
        raise ValueError("DeepSeek-V4 artifact must be a dictionary")
    keys = (
        "prefill_coeff",
        "prefill_selection_loss",
        "prefill_confidence_loss",
        "decode_coeff",
        "decode_loss",
    )
    missing = [key for key in (*keys, "norms") if key not in payload]
    if missing:
        raise ValueError(f"DeepSeek-V4 artifact is missing keys: {missing}")
    if any(tuple(payload[key].shape) != shape for key in keys):
        raise ValueError("DeepSeek-V4 matrix shape mismatch")
    if tuple(payload["norms"].shape) != shape[:2]:
        raise ValueError("DeepSeek-V4 norm shape mismatch")
    for key in (*keys, "norms"):
        if not torch.isfinite(torch.as_tensor(payload[key])).all():
            raise ValueError(f"DeepSeek-V4 {key} contains non-finite values")

    return {
        key: _loss_health(key, payload[key])
        for key in (
            "prefill_selection_loss",
            "prefill_confidence_loss",
            "decode_loss",
        )
    }


def validate(path: Path) -> dict[str, dict[str, float]]:
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
        return {}
    elif path.name == "deepseek-v4-flash.pt":
        return validate_deepseek_payload(payload)
    else:
        raise ValueError(f"Unknown artifact: {path.name}")
