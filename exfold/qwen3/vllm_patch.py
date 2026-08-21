"""Qwen3-MoE vLLM routing patch for ExFold."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from exfold.qwen3.kernels import (
    project_prefill_topk,
    remap_decode_static,
    select_and_reroute_decode,
)


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _layer_index(prefix: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else None


def _prefix(args: tuple, kwargs: dict) -> str:
    if isinstance(kwargs.get("prefix"), str):
        return kwargs["prefix"]
    return next((value for value in reversed(args) if isinstance(value, str)), "")


@lru_cache(maxsize=1)
def _load_artifact(path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"coeff", "loss", "expert_norms"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"artifact must contain {sorted(required)}")

    coefficient = torch.as_tensor(payload["coeff"], dtype=torch.float32).contiguous()
    loss = torch.as_tensor(payload["loss"], dtype=torch.float32).contiguous()
    norms = torch.as_tensor(payload["expert_norms"], dtype=torch.float32).contiguous()
    if coefficient.ndim != 3 or coefficient.shape[1] != coefficient.shape[2]:
        raise ValueError("coeff must have shape [layers, experts, experts]")
    if loss.shape != coefficient.shape:
        raise ValueError("loss and coeff must have identical shapes")
    if norms.shape != coefficient.shape[:2]:
        raise ValueError("expert_norms must have shape [layers, experts]")
    if not all(torch.isfinite(item).all() for item in (coefficient, loss, norms)):
        raise ValueError("artifact contains non-finite values")
    if torch.any(loss < 0) or torch.any(norms <= 0):
        raise ValueError("loss must be non-negative and expert_norms positive")

    coefficient = coefficient.masked_fill(loss >= 1e30, 0.0)
    coefficient.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    loss.diagonal(dim1=-2, dim2=-1).fill_(1e30)
    return coefficient, loss, norms


def _phase(tokens: int) -> str:
    forced = os.environ.get("EXFOLD_FORCE_PHASE", "").strip().lower()
    if forced:
        if forced not in {"prefill", "decode"}:
            raise ValueError("EXFOLD_FORCE_PHASE must be 'prefill' or 'decode'")
        return forced

    from vllm.forward_context import get_forward_context

    metadata = get_forward_context().attn_metadata
    if isinstance(metadata, dict):
        metadata = next(
            (
                item
                for item in metadata.values()
                if hasattr(item, "num_prefill_tokens")
            ),
            None,
        )
    prefill = getattr(metadata, "num_prefill_tokens", None)
    decode = getattr(metadata, "num_decode_tokens", None)
    if prefill is None or decode is None:
        raise RuntimeError("vLLM attention metadata is unavailable")
    prefill, decode = int(prefill), int(decode)
    if prefill + decode != tokens:
        raise RuntimeError("MoE token count disagrees with vLLM attention metadata")
    if prefill and decode:
        raise RuntimeError(
            "mixed prefill/decode batches are unsupported; disable chunked prefill"
        )
    return "prefill" if prefill else "decode"


def _native_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk

    weights, expert_ids, _ = fused_topk(
        hidden_states, gating_output, topk, renormalize
    )
    return weights, expert_ids


def _make_router(
    coefficient_cpu: torch.Tensor,
    loss_cpu: torch.Tensor,
    norms_cpu: torch.Tensor,
    prefill_budget: int,
    decode_budget: int,
) -> Callable:
    decode_policy = os.environ.get("EXFOLD_QWEN3_DECODE_POLICY", "dynamic").lower()
    if decode_policy not in {"dynamic", "static_norm"}:
        raise ValueError(
            "EXFOLD_QWEN3_DECODE_POLICY must be 'dynamic' or 'static_norm'"
        )
    debug_routing = os.environ.get("EXFOLD_DEBUG_ROUTING") == "1"
    debug_emitted = False
    device_cache: dict[
        torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = {}
    static_cpu: tuple[torch.Tensor, torch.Tensor] | None = None
    if decode_policy == "static_norm" and decode_budget < coefficient_cpu.shape[0]:
        retained = torch.topk(norms_cpu, k=decode_budget, sorted=False).indices.long()
        candidate_loss = loss_cpu[:, retained]
        target_position = candidate_loss.argmin(dim=1)
        target = retained[target_position]
        scale = coefficient_cpu[
            torch.arange(coefficient_cpu.shape[0]), target
        ].float()
        is_retained = torch.zeros(coefficient_cpu.shape[0], dtype=torch.bool)
        is_retained[retained] = True
        source = torch.arange(coefficient_cpu.shape[0])
        target = torch.where(is_retained, source, target).to(torch.int32)
        scale = torch.where(is_retained, torch.ones_like(scale), scale)
        static_cpu = target.contiguous(), scale.contiguous()
    static_device_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def tensors_for(
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = device_cache.get(device)
        if cached is None:
            cached = (
                coefficient_cpu.to(device),
                loss_cpu.to(device),
                norms_cpu.to(device),
            )
            device_cache[device] = cached
        return cached

    def router(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal debug_emitted
        weights, expert_ids = _native_topk(
            hidden_states, gating_output, topk, renormalize
        )
        coefficient, loss, norms = tensors_for(gating_output.device)

        if _phase(gating_output.shape[0]) == "prefill":
            target_k = min(prefill_budget, topk)
            if target_k < topk:
                weights, expert_ids = project_prefill_topk(
                    weights, expert_ids, coefficient, target_k, loss
                )
        else:
            num_experts = gating_output.shape[1]
            subset_size = min(decode_budget, num_experts)
            if subset_size < num_experts:
                before = (
                    int(torch.unique(expert_ids).numel())
                    if debug_routing and not debug_emitted
                    else 0
                )
                if decode_policy == "static_norm":
                    cached = static_device_cache.get(gating_output.device)
                    if cached is None:
                        assert static_cpu is not None
                        cached = tuple(
                            tensor.to(gating_output.device) for tensor in static_cpu
                        )
                        static_device_cache[gating_output.device] = cached
                    weights, expert_ids = remap_decode_static(
                        weights, expert_ids, cached[0], cached[1]
                    )
                else:
                    weights, expert_ids = select_and_reroute_decode(
                        weights,
                        expert_ids,
                        norms,
                        coefficient,
                        loss,
                        subset_size,
                    )
                if debug_routing and not debug_emitted:
                    after = int(torch.unique(expert_ids).numel())
                    if before > subset_size:
                        print(
                            "[ExFold][routing-debug] "
                            f"tokens={gating_output.shape[0]} "
                            f"before_unique={before} selected={subset_size} "
                            f"after_unique={after}",
                            flush=True,
                        )
                        debug_emitted = True
        return weights.float(), expert_ids.int()

    return router


def patch_vllm_qwen3_moe() -> None:
    """Install ExFold before vLLM constructs Qwen3 MoE layers."""
    artifact_path = os.environ.get("EXFOLD_ARTIFACT")
    if not artifact_path:
        raise ValueError("EXFOLD_ARTIFACT is required")
    coefficient, loss, norms = _load_artifact(str(Path(artifact_path).resolve()))
    prefill_budget = _positive_int("EXFOLD_PREFILL_BUDGET", 8)
    decode_budget = _positive_int("EXFOLD_DECODE_BUDGET", coefficient.shape[1])

    from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock

    if getattr(Qwen3MoeSparseMoeBlock, "_exfold_patched", False):
        return
    original_init = Qwen3MoeSparseMoeBlock.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        layer = _layer_index(_prefix(args, kwargs))
        if layer is None:
            raise ValueError("unable to infer the Qwen3 MoE layer index")
        if layer >= coefficient.shape[0]:
            raise ValueError(f"artifact does not contain layer {layer}")
        if coefficient.shape[1] != self.n_routed_experts:
            raise ValueError("artifact expert count does not match the model")
        self.experts.custom_routing_function = _make_router(
            coefficient[layer],
            loss[layer],
            norms[layer],
            prefill_budget,
            decode_budget,
        )

    Qwen3MoeSparseMoeBlock.__init__ = patched_init
    Qwen3MoeSparseMoeBlock._exfold_patched = True
    print(
        "[ExFold] installed "
        f"prefill_budget={prefill_budget} decode_budget={decode_budget} "
        f"decode_policy={os.environ.get('EXFOLD_QWEN3_DECODE_POLICY', 'dynamic')}",
        flush=True,
    )
