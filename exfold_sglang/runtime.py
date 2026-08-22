from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from exfold_sglang.calibration import load_calibration


_INSTALLED = False


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _prepare_layer(topk: Any, layer_id: int) -> None:
    artifact = Path(os.environ["EXFOLD_CALIBRATION_PATH"]).resolve()
    calibration = load_calibration(str(artifact))
    device = torch.device("cuda", torch.cuda.current_device())
    topk._exfold_sglang = True
    topk._exfold_layer_id = layer_id
    topk._exfold_phase = "decode"
    topk._exfold_trace_done = set()
    topk._exfold_calibration = tuple(
        tensor[layer_id].to(device=device, non_blocking=False).contiguous()
        for tensor in calibration
    )


def _fold_output(topk: Any, output: Any):
    from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutputChecker

    if not TopKOutputChecker.format_is_standard(output):
        raise RuntimeError(
            "ExFold requires standard top-k tensors; use --moe-runner-backend marlin"
        )

    weights, ids, router_logits = output
    if weights.shape[0] == 0:
        return output

    topk_config = getattr(topk, "topk_config", None)
    num_shared = int(
        topk_config.num_fused_shared_experts
        if topk_config is not None
        else getattr(topk, "num_fused_shared_experts", 0)
    )
    routed_width = weights.shape[1] - num_shared
    if routed_width <= 0:
        return output
    routed_weights = weights[:, :routed_width]
    routed_ids = ids[:, :routed_width]
    shared_weights = weights[:, routed_width:]
    shared_ids = ids[:, routed_width:]

    (
        prefill_coeff,
        prefill_loss,
        confidence_loss,
        decode_coeff,
        decode_loss,
        norms,
    ) = topk._exfold_calibration
    prefill_k = _env_int("EXFOLD_PREFILL_TOPK", 4)
    decode_budget = _env_int("EXFOLD_DECODE_EXPERT_BUDGET", 128)
    phase = topk._exfold_phase

    if phase == "prefill" and prefill_k < routed_width:
        from exfold_sglang.ops import fold_prefill

        routed_weights, routed_ids = fold_prefill(
            routed_weights,
            routed_ids,
            prefill_coeff,
            prefill_loss,
            confidence_loss,
            prefill_k,
            2,
        )
    elif phase == "decode" and decode_budget < 256:
        # If there cannot be more active experts than the budget, remapping is
        # an exact no-op and skipping the custom kernel is faster.
        if routed_weights.shape[0] * routed_width > decode_budget:
            from exfold_sglang.ops import fold_decode_score

            routed_weights, routed_ids = fold_decode_score(
                routed_weights,
                routed_ids,
                decode_coeff,
                decode_loss,
                norms,
                decode_budget,
                min(prefill_k, routed_width),
            )

    if num_shared:
        weights = torch.cat((routed_weights, shared_weights), dim=1)
        ids = torch.cat((routed_ids, shared_ids), dim=1)
    else:
        weights, ids = routed_weights, routed_ids

    if topk._exfold_layer_id == 3 and phase not in topk._exfold_trace_done:
        topk._exfold_trace_done.add(phase)
        print(
            "[ExFold SGLang] "
            f"phase={phase} routed_topk={routed_width}->{routed_weights.shape[1]} "
            f"decode_budget={decode_budget} importance=s_i shared={num_shared}",
            flush=True,
        )
    return StandardTopKOutput(weights, ids, router_logits)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("EXFOLD_SGLANG_ENABLE=1 requires CUDA")
    if not os.environ.get("EXFOLD_CALIBRATION_PATH"):
        raise RuntimeError("EXFOLD_CALIBRATION_PATH is required")

    # Importing the classes is sufficient; DeepSeek-V4 reuses DeepseekV2MoE.
    from sglang.srt.layers.moe.hash_topk import HashTopK
    from sglang.srt.layers.moe.topk import TopK
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

    original_init = DeepseekV2MoE.__init__
    original_forward = DeepseekV2MoE.forward
    original_topk_forward = TopK.forward

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        is_deepseek_v4 = bool(kwargs.get("is_deepseek_v4", False))
        if not is_deepseek_v4 and len(args) >= 7:
            is_deepseek_v4 = bool(args[6])
        if not is_deepseek_v4:
            return
        layer_id = int(kwargs.get("layer_id", args[1] if len(args) > 1 else self.layer_id))
        if isinstance(self.topk, HashTopK):
            # DSV4's first three layers use checkpoint-defined hash routing.
            # Keep those layers bit-for-bit identical to the original model.
            self._exfold_sglang = False
            return
        _prepare_layer(self.topk, layer_id)
        self._exfold_sglang = True

    def patched_forward(self, hidden_states, forward_batch=None, *args, **kwargs):
        if getattr(self, "_exfold_sglang", False):
            mode = getattr(forward_batch, "forward_mode", None)
            is_decode = bool(
                mode is not None
                and (mode.is_decode() or getattr(mode, "is_target_verify", lambda: False)())
            )
            self.topk._exfold_phase = "decode" if is_decode else "prefill"
        return original_forward(self, hidden_states, forward_batch, *args, **kwargs)

    def patched_topk_forward(self, *args, **kwargs):
        # MultiPlatformOp caches the selected platform implementation on each
        # instance. Wrap the public dispatch point so CUDA, native, and compiled
        # paths all pass through the same ExFold transform.
        output = original_topk_forward(self, *args, **kwargs)
        if not getattr(self, "_exfold_sglang", False):
            return output
        return _fold_output(self, output)

    DeepseekV2MoE.__init__ = patched_init
    DeepseekV2MoE.forward = patched_forward
    TopK.forward = patched_topk_forward
    _INSTALLED = True
    print(
        "[ExFold SGLang] installed P"
        f"{_env_int('EXFOLD_PREFILL_TOPK', 4)}+D"
        f"{_env_int('EXFOLD_DECODE_EXPERT_BUDGET', 128)} runtime "
        "(decode importance=s_i)",
        flush=True,
    )
