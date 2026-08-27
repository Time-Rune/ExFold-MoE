"""Runtime routing patch for DeepSeek-V4-Flash in vLLM.

The patch is intentionally activated only by DEEPSEEK_V4_RUNTIME_ENABLE=1.
This keeps manifest/config helper processes lightweight and leaves the original
model path byte-for-byte equivalent to upstream vLLM.
"""

import os
import re
import sys
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Optional

def _enabled() -> bool:
    if os.environ.get("DEEPSEEK_V4_RUNTIME_ENABLE", "0") != "1":
        return False
    original_argv = " ".join(getattr(sys, "orig_argv", sys.argv))
    return "multiprocessing.resource_tracker" not in original_argv


def _layer_index(prefix: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", prefix)
    if match is None:
        raise ValueError(f"Cannot parse DeepSeek-V4 layer index from {prefix!r}")
    return int(match.group(1))


def _phase_token_counts(
    num_tokens: int,
    unavailable: Optional[tuple[int, int]] = None,
) -> tuple[int, int]:
    from vllm.forward_context import get_forward_context

    metadata = _find_phase_metadata(get_forward_context().attn_metadata)
    prefill = getattr(metadata, "num_prefill_tokens", None)
    decode = getattr(metadata, "num_decode_tokens", None)
    if prefill is None or decode is None:
        if unavailable is not None:
            return unavailable
        raise RuntimeError("vLLM attention metadata is unavailable")
    prefill, decode = int(prefill), int(decode)
    if prefill + decode != num_tokens:
        raise RuntimeError(
            f"MoE token mismatch: prefill={prefill}, decode={decode}, total={num_tokens}"
        )
    return decode, prefill


def _find_phase_metadata(metadata):
    """Find token counts in vLLM's nested per-backend metadata containers."""
    pending = [metadata]
    visited: set[int] = set()
    while pending:
        item = pending.pop()
        if item is None or id(item) in visited:
            continue
        visited.add(id(item))
        if hasattr(item, "num_prefill_tokens") and hasattr(
            item, "num_decode_tokens"
        ):
            return item
        if isinstance(item, Mapping):
            pending.extend(item.values())
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            pending.extend(item)
    return None


@lru_cache(maxsize=1)
def _load_calibration(path: str):
    import torch
    from exfold.artifacts import validate_deepseek_payload

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Calibration artifact must be a dictionary")
    validate_deepseek_payload(payload)
    prefill_coeff = torch.as_tensor(
        payload.get("prefill_coeff", payload.get("coeff")), dtype=torch.float32
    ).contiguous()
    prefill_loss = torch.as_tensor(
        payload.get(
            "prefill_selection_loss",
            payload.get("selection_loss", payload.get("loss")),
        ),
        dtype=torch.float32,
    ).clone().contiguous()
    prefill_confidence_loss = torch.as_tensor(
        payload.get("prefill_confidence_loss", prefill_loss),
        dtype=torch.float32,
    ).clone().contiguous()
    decode_coeff = torch.as_tensor(
        payload.get(
            "decode_coeff", payload.get("routing_coeff", payload.get("coeff"))
        ),
        dtype=torch.float32,
    ).contiguous()
    decode_loss = torch.as_tensor(
        payload.get(
            "decode_loss", payload.get("routing_loss", payload.get("loss"))
        ),
        dtype=torch.float32,
    ).contiguous()
    norms_payload = payload.get(
        "decode_norms", payload.get("routing_norms", payload.get("norms"))
    )
    if prefill_coeff.ndim != 3 or prefill_coeff.shape[1] != prefill_coeff.shape[2]:
        raise ValueError(
            "prefill_coeff must have shape [layers, experts, experts], "
            f"got {prefill_coeff.shape}"
        )
    for name, tensor in (
        ("prefill_loss", prefill_loss),
        ("prefill_confidence_loss", prefill_confidence_loss),
        ("decode_coeff", decode_coeff),
        ("decode_loss", decode_loss),
    ):
        if tensor.shape != prefill_coeff.shape:
            raise ValueError(
                f"{name} shape {tensor.shape} does not match {prefill_coeff.shape}"
            )
    if norms_payload is None:
        raise ValueError("Calibration artifact is missing norms")
    if isinstance(norms_payload, dict):
        norms = torch.stack(
            [
                torch.as_tensor(norms_payload[i], dtype=torch.float32)
                for i in range(prefill_coeff.shape[0])
            ]
        )
    else:
        norms = torch.as_tensor(norms_payload, dtype=torch.float32)
    if norms.shape != prefill_coeff.shape[:2]:
        raise ValueError(
            f"norms must have shape {prefill_coeff.shape[:2]}, got {norms.shape}"
        )
    for name, tensor in (
        ("prefill_coeff", prefill_coeff),
        ("prefill_loss", prefill_loss),
        ("prefill_confidence_loss", prefill_confidence_loss),
        ("decode_coeff", decode_coeff),
        ("decode_loss", decode_loss),
    ):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Calibration {name} contains non-finite values")
    # The method always assigns an omitted source to the minimum-loss retained
    # target. Only the 1e30 sentinel denotes an unavailable pair; a loss above
    # one is still a valid (albeit weak) projection. Keep an environment
    # override for controlled rejection-threshold ablations.
    max_projection_loss = float(
        os.environ.get("EXFOLD_MAX_PROJECTION_LOSS", "1e30")
    )
    decode_loss.masked_fill_(decode_loss > max_projection_loss, 1.0e30)
    prefill_coeff.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    prefill_loss.diagonal(dim1=-2, dim2=-1).fill_(1.0e30)
    prefill_confidence_loss.diagonal(dim1=-2, dim2=-1).fill_(0.0)
    decode_coeff.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    decode_loss.diagonal(dim1=-2, dim2=-1).fill_(1.0e30)
    return (
        prefill_coeff,
        prefill_loss,
        prefill_confidence_loss,
        decode_coeff,
        decode_loss,
        norms.clamp_min(torch.finfo(torch.float32).eps).contiguous(),
    )


_INPUT_IDS_UNSET = object()
_CORRECTION_BIAS_UNSET = object()
def _native_route(
    module,
    hidden_states,
    router_logits,
    topk: int,
    input_ids=_INPUT_IDS_UNSET,
    correction_bias=_CORRECTION_BIAS_UNSET,
):
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        fused_topk_bias,
    )

    if input_ids is _INPUT_IDS_UNSET:
        input_ids = getattr(module, "_dsv4_runtime_input_ids", None)
    if module.gate.tid2eid is not None and input_ids is None:
        raise RuntimeError("Hash-MoE routing requires input_ids")
    if correction_bias is _CORRECTION_BIAS_UNSET:
        correction_bias = (
            module.gate.e_score_correction_bias.data
            if module.gate.e_score_correction_bias is not None
            else None
        )
    weights, ids = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=router_logits,
        scoring_func=module.scoring_func,
        e_score_correction_bias=correction_bias,
        topk=topk,
        renormalize=module.renormalize,
        indices_type=module.hash_indices_dtype,
        input_tokens=input_ids,
        hash_indices_table=module.gate.tid2eid,
        routed_scaling_factor=module.routed_scaling_factor,
    )
    return weights, ids


def _make_router(
    module,
    layer: int,
    calibration,
):
    import torch

    from exfold.deepseek_v4.cuda_ops import fold_decode, fold_prefill

    original_topk = int(module.n_activated_experts)
    prefill_topk = int(os.environ.get("EXFOLD_PREFILL_TOPK", original_topk))
    prefill_renorm_name = os.environ.get(
        "EXFOLD_PREFILL_RENORM", "none"
    ).strip().lower()
    prefill_renorm_modes = {"none": 0, "router_l2": 1, "similarity_blend": 2}
    if prefill_renorm_name not in prefill_renorm_modes:
        raise ValueError(
            "EXFOLD_PREFILL_RENORM must be none, router_l2, or similarity_blend"
        )
    prefill_renorm_mode = prefill_renorm_modes[prefill_renorm_name]
    decode_budget = int(os.environ.get("EXFOLD_DECODE_EXPERT_BUDGET", 256))
    device_cache: dict[torch.device, tuple[torch.Tensor, ...]] = {}
    phase_trace_calls = 0

    def calibration_for(device: torch.device):
        cached = device_cache.get(device)
        if cached is None:
            if calibration is None:
                raise RuntimeError("ExFold routing requires calibration tensors")
            (
                prefill_coeff,
                prefill_loss,
                prefill_confidence_loss,
                decode_coeff,
                decode_loss,
                decode_norms,
            ) = calibration
            cached = (
                prefill_coeff[layer].to(device=device, non_blocking=True),
                prefill_loss[layer].to(device=device, non_blocking=True),
                prefill_confidence_loss[layer].to(
                    device=device, non_blocking=True
                ),
                decode_coeff[layer].to(device=device, non_blocking=True),
                decode_loss[layer].to(device=device, non_blocking=True),
                decode_norms[layer].to(device=device, non_blocking=True),
            )
            device_cache[device] = cached
        return cached

    def route_one(hidden_states, gating_output, phase: str, input_ids):
        weights, ids = _native_route(
            module,
            hidden_states,
            gating_output,
            original_topk,
            input_ids=input_ids,
        )
        if phase == "prefill":
            if prefill_topk >= original_topk:
                return weights, ids
            prefill_coeff, prefill_loss, prefill_confidence_loss, _, _, _ = calibration_for(
                gating_output.device
            )
            return fold_prefill(
                weights,
                ids,
                prefill_coeff,
                prefill_loss,
                prefill_confidence_loss,
                prefill_topk,
                prefill_renorm_mode,
            )

        # With no more route slots than the budget, every active expert is
        # retained. This static-shape fast path is exact and graph-safe.
        max_active_experts = weights.shape[0] * weights.shape[1]
        if (
            decode_budget >= module.n_routed_experts
            or max_active_experts <= decode_budget
        ):
            return weights, ids
        _, _, _, coeff, loss, norms = calibration_for(gating_output.device)
        return fold_decode(
            weights,
            ids,
            coeff,
            loss,
            norms,
            decode_budget,
        )

    def router(hidden_states, gating_output, topk, renormalize, input_ids=None):
        nonlocal phase_trace_calls
        del topk, renormalize
        num_tokens = int(gating_output.shape[0])
        num_decode, num_prefill = _phase_token_counts(
            num_tokens, unavailable=(0, num_tokens)
        )
        if (
            layer == 0
            and os.environ.get("EXFOLD_RUNTIME_TRACE", "0") == "1"
            and phase_trace_calls
            < int(os.environ.get("EXFOLD_RUNTIME_TRACE_PHASE_CALLS", "24"))
        ):
            print(
                "[DeepSeek-V4 phase trace] "
                f"call={phase_trace_calls} total={num_tokens} "
                f"decode={num_decode} prefill={num_prefill}",
                flush=True,
            )
            phase_trace_calls += 1
        if not (num_decode and num_prefill):
            phase = "prefill" if num_prefill else "decode"
            return route_one(hidden_states, gating_output, phase, input_ids)

        decode_input_ids = input_ids[:num_decode] if input_ids is not None else None
        prefill_input_ids = input_ids[num_decode:] if input_ids is not None else None
        decode_weights, decode_ids = route_one(
            hidden_states[:num_decode],
            gating_output[:num_decode],
            "decode",
            decode_input_ids,
        )
        prefill_weights, prefill_ids = route_one(
            hidden_states[num_decode:],
            gating_output[num_decode:],
            "prefill",
            prefill_input_ids,
        )
        width = max(decode_ids.shape[1], prefill_ids.shape[1])

        def pad(weights, ids):
            missing = width - ids.shape[1]
            if missing <= 0:
                return weights, ids
            return (
                torch.cat(
                    [
                        weights,
                        torch.zeros(
                            (weights.shape[0], missing),
                            dtype=weights.dtype,
                            device=weights.device,
                        ),
                    ],
                    dim=1,
                ),
                torch.cat([ids, ids[:, -1:].expand(-1, missing)], dim=1),
            )

        decode_weights, decode_ids = pad(decode_weights, decode_ids)
        prefill_weights, prefill_ids = pad(prefill_weights, prefill_ids)
        return (
            torch.cat([decode_weights, prefill_weights], dim=0),
            torch.cat([decode_ids, prefill_ids], dim=0),
        )

    router._dsv4_accepts_input_ids = True
    return router


def _install() -> None:
    import torch

    if os.environ.get("DEEPSEEK_V4_ROUTING_MODE", "exfold") != "exfold":
        raise ValueError("The public DeepSeek-V4 runtime supports only ExFold")
    calibration_path = Path(os.environ["EXFOLD_CALIBRATION_PATH"]).resolve()
    calibration = _load_calibration(str(calibration_path))

    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
        CustomRoutingRouter,
    )
    try:
        from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MoE
    except ImportError:
        from vllm.model_executor.models.deepseek_v4 import DeepseekV4MoE

    if getattr(DeepseekV4MoE, "_dsv4_runtime_patched", False):
        return

    if not getattr(CustomRoutingRouter, "_dsv4_input_ids_patched", False):
        original_compute_routing = CustomRoutingRouter._compute_routing

        def patched_compute_routing(
            self,
            hidden_states,
            router_logits,
            indices_type,
            *,
            input_ids=None,
        ):
            if not getattr(
                self.custom_routing_function, "_dsv4_accepts_input_ids", False
            ):
                return original_compute_routing(
                    self,
                    hidden_states,
                    router_logits,
                    indices_type,
                    input_ids=input_ids,
                )
            topk_weights, topk_ids = self.custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
                input_ids=input_ids,
            )
            return topk_weights.to(torch.float32), topk_ids.to(
                torch.int32 if indices_type is None else indices_type
            )

        CustomRoutingRouter._compute_routing = patched_compute_routing
        CustomRoutingRouter._dsv4_input_ids_patched = True

    original_init_fused = DeepseekV4MoE._init_fused_moe_experts

    def patched_init_fused(self, *args, **kwargs):
        if kwargs:
            import inspect

            bound = inspect.signature(original_init_fused).bind(
                self, *args, **kwargs
            )
            vllm_config = bound.arguments.get("vllm_config")
            config = bound.arguments["config"]
            quant_config = bound.arguments["quant_config"]
            prefix = bound.arguments["prefix"]
        elif len(args) == 3:
            vllm_config = None
            config, quant_config, prefix = args
        elif len(args) == 4:
            vllm_config, config, quant_config, prefix = args
        else:
            raise TypeError(
                "Unsupported DeepseekV4MoE._init_fused_moe_experts "
                f"signature: {len(args)} positional arguments"
            )

        # Initialize the TP/EP metadata without constructing the upstream
        # runner first. vLLM >= 0.26 registers every MoE runner by layer name,
        # so constructing and then replacing it raises a duplicate-name error.
        from vllm.distributed import get_tensor_model_parallel_rank

        self.tp_rank = get_tensor_model_parallel_rank()
        if vllm_config is None:
            # Legacy DeepSeek-V4 path did not support EPLB here.
            self.n_redundant_experts = 0
        else:
            self.n_redundant_experts = (
                vllm_config.parallel_config.eplb_config.num_redundant_experts
            )
        self.n_shared_experts = config.n_shared_experts or 0
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = (
            self.n_logical_experts + self.n_redundant_experts
        )
        if self.n_physical_experts % self.tp_size != 0:
            raise ValueError(
                f"n_physical_experts={self.n_physical_experts} must be divisible "
                f"by tp_size={self.tp_size}"
            )
        self.n_local_physical_experts = self.n_physical_experts // self.tp_size
        self.n_local_experts = self.n_local_physical_experts
        self.experts_start_idx = self.tp_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.physical_expert_start = self.experts_start_idx
        self.physical_expert_end = self.experts_end_idx

        layer = _layer_index(prefix)
        router = _make_router(self, layer, calibration)
        parallel_config = (
            vllm_config.parallel_config if vllm_config is not None else None
        )
        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            # DeepSeek-V4's V1 runner keeps the gate inside the MoE custom op.
            # Preserving that path avoids an external gate launch on every
            # layer while still letting CustomRoutingRouter fold its output.
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=int(config.num_experts_per_tok),
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=self.gate.tid2eid,
            swiglu_limit=self.swiglu_limit,
            router_logits_dtype=torch.float32,
            custom_routing_function=router,
            enable_eplb=(
                parallel_config.enable_eplb
                if parallel_config is not None
                else False
            ),
            num_redundant_experts=getattr(self, "n_redundant_experts", 0),
        )
        if layer == 0:
            print(
                "[DeepSeek-V4 runtime] "
                f"internal_router={getattr(self.experts, 'is_internal_router', True)}",
                flush=True,
            )

    DeepseekV4MoE._init_fused_moe_experts = patched_init_fused
    DeepseekV4MoE._dsv4_runtime_patched = True
    print(
        "[DeepSeek-V4 runtime] installed ExFold routing",
        flush=True,
    )


if _enabled():
    _install()
