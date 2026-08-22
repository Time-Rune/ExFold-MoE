from pathlib import Path

import torch


def _load_extension() -> None:
    library = Path(__file__).resolve().parent / "cuda_ops/build/libdsv4_exfold_cuda.so"
    if not library.is_file():
        raise RuntimeError(
            f"ExFold CUDA extension is missing: {library}. "
            "Run `python -m exfold_sglang.cuda_ops.build` first."
        )
    torch.ops.load_library(str(library))


_load_extension()


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
    protected_topk: int,
    budget: int,
    mode: int,
    importance_mode: int,
):
    del coefficient, loss, norms, protected_topk, budget, mode, importance_mode
    return torch.empty_like(weights), torch.empty_like(ids)


@torch.library.register_fake("dsv4_exfold::decode_compact")
def _decode_compact_fake(
    weights,
    ids,
    coefficient,
    loss,
    norms,
    protected_topk: int,
    budget: int,
    importance_mode: int,
):
    del coefficient, loss, norms, budget, importance_mode
    return (
        weights.new_empty((weights.shape[0], protected_topk)),
        ids.new_empty((ids.shape[0], protected_topk)),
    )


@torch.library.register_fake("dsv4_exfold::mhc_sinkhorn")
def _mhc_sinkhorn_fake(
    mixes,
    hc_scale,
    hc_base,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
):
    del hc_scale, hc_base, sinkhorn_iters, eps
    leading = tuple(mixes.shape[:-1])
    return (
        mixes.new_empty((*leading, hc_mult)),
        mixes.new_empty((*leading, hc_mult)),
        mixes.new_empty((*leading, hc_mult, hc_mult)),
    )


def fold_prefill(
    weights,
    ids,
    coefficient,
    selection_loss,
    confidence_loss,
    target_k: int,
    renorm_mode: int,
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


def fold_decode_score(
    weights,
    ids,
    coefficient,
    loss,
    norms,
    budget: int,
    protected_topk: int,
):
    # mode=0 uses all routed slots. importance_mode=1 ranks by sum(router score).
    return torch.ops.dsv4_exfold.decode(
        weights.contiguous(),
        ids.contiguous(),
        coefficient,
        loss,
        norms,
        protected_topk,
        budget,
        0,
        1,
    )


def mhc_split_sinkhorn(
    mixes,
    hc_scale,
    hc_base,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1.0e-6,
):
    return torch.ops.dsv4_exfold.mhc_sinkhorn(
        mixes.contiguous(),
        hc_scale.contiguous(),
        hc_base.contiguous(),
        hc_mult,
        sinkhorn_iters,
        eps,
    )
