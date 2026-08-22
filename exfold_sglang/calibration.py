from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def load_calibration(path: str) -> tuple[torch.Tensor, ...]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("ExFold calibration artifact must be a dictionary")

    prefill_coeff = torch.as_tensor(
        payload.get("prefill_coeff", payload.get("coeff")), dtype=torch.float32
    ).contiguous()
    prefill_loss = torch.as_tensor(
        payload.get(
            "prefill_selection_loss",
            payload.get("selection_loss", payload.get("loss")),
        ),
        dtype=torch.float32,
    ).contiguous()
    confidence_loss = torch.as_tensor(
        payload.get("prefill_confidence_loss", prefill_loss), dtype=torch.float32
    ).contiguous()
    decode_coeff = torch.as_tensor(
        payload.get("decode_coeff", payload.get("coeff")), dtype=torch.float32
    ).contiguous()
    decode_loss = torch.as_tensor(
        payload.get("decode_loss", payload.get("loss")), dtype=torch.float32
    ).contiguous()
    norms = torch.as_tensor(
        payload.get("decode_norms", payload.get("norms")), dtype=torch.float32
    ).contiguous()

    expected = (43, 256, 256)
    for name, tensor in (
        ("prefill_coeff", prefill_coeff),
        ("prefill_loss", prefill_loss),
        ("prefill_confidence_loss", confidence_loss),
        ("decode_coeff", decode_coeff),
        ("decode_loss", decode_loss),
    ):
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")
    if tuple(norms.shape) != expected[:2]:
        raise ValueError(f"norms must have shape {expected[:2]}, got {tuple(norms.shape)}")

    return (
        prefill_coeff,
        prefill_loss,
        confidence_loss,
        decode_coeff,
        decode_loss,
        norms.clamp_min(torch.finfo(torch.float32).eps),
    )
