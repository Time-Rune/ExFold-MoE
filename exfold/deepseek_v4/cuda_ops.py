from __future__ import annotations

import os
from pathlib import Path

import torch


def _load() -> None:
    library = os.environ.get("EXFOLD_CUDA_LIBRARY")
    if not library:
        library = str(
            Path(__file__).resolve().parent
            / "csrc"
            / "build"
            / "libdsv4_exfold_cuda.so"
        )
    if not Path(library).is_file():
        raise RuntimeError(f"ExFold CUDA library is missing: {library}")
    torch.ops.load_library(library)
    print(f"[ExFold CUDA] loaded custom op: {library}", flush=True)


_load()


@torch.library.register_fake("dsv4_exfold::prefill")
def _prefill_fake(
    weights,
    ids,
    coefficient,
    selection_loss,
    confidence_loss,
    target_k: int,
    renorm_mode: int,
):
    del coefficient, selection_loss, confidence_loss, renorm_mode
    return (
        weights.new_empty((weights.shape[0], target_k)),
        ids.new_empty((ids.shape[0], target_k)),
    )


@torch.library.register_fake("dsv4_exfold::decode")
def _decode_fake(
    weights,
    ids,
    coefficient,
    loss,
    norms,
    budget: int,
):
    del coefficient, loss, norms, budget
    return torch.empty_like(weights), torch.empty_like(ids)


def fold_prefill(
    weights,
    ids,
    coefficient,
    selection_loss,
    confidence_loss,
    target_k: int,
    renorm_mode: int = 0,
):
    return torch.ops.dsv4_exfold.prefill(
        weights.contiguous(),
        ids.contiguous(),
        coefficient,
        selection_loss,
        confidence_loss,
        target_k,
        renorm_mode,
    )


def fold_decode(
    weights,
    ids,
    coefficient,
    loss,
    norms,
    budget: int,
):
    return torch.ops.dsv4_exfold.decode(
        weights.contiguous(),
        ids.contiguous(),
        coefficient,
        loss,
        norms,
        budget,
    )
