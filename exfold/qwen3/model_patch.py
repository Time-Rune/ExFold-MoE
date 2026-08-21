"""Minimal Qwen3-MoE forward patch for collecting expert outputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from exfold.qwen3.calibration import PairwiseScalarStats


def prepare_qwen3_moe():
    """Patch only the sparse MoE aggregation used during calibration."""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeForCausalLM,
        Qwen3MoeSparseMoeBlock,
    )

    if getattr(Qwen3MoeSparseMoeBlock, "_exfold_calibration_patched", False):
        return Qwen3MoeForCausalLM, Qwen3MoeSparseMoeBlock

    def calibration_forward(self, hidden_states: torch.Tensor):
        original_shape = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        router_logits = self.gate(flat_hidden)
        router_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        router_weights, expert_ids = torch.topk(router_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_weights /= router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(flat_hidden.dtype)

        expert_outputs = flat_hidden.new_zeros(
            expert_ids.shape + (flat_hidden.shape[-1],)
        )
        for expert_index in range(self.num_experts):
            token_ids, slot_ids = (expert_ids == expert_index).nonzero(as_tuple=True)
            if token_ids.numel() == 0:
                continue
            expert_outputs[token_ids, slot_ids] = self.experts[expert_index](
                flat_hidden[token_ids]
            )

        output = (
            expert_outputs * router_weights.unsqueeze(-1)
        ).sum(dim=1).reshape(original_shape)
        self.exfold_stats.update(expert_ids, expert_outputs)
        return output, router_logits

    Qwen3MoeSparseMoeBlock.forward = calibration_forward
    Qwen3MoeSparseMoeBlock._exfold_calibration_patched = True
    return Qwen3MoeForCausalLM, Qwen3MoeSparseMoeBlock


def attach_stats(
    model: torch.nn.Module,
    moe_class: type,
    ridge: float,
    clip_min: float,
    clip_max: float,
) -> list[PairwiseScalarStats]:
    """Attach one statistics accumulator to every Qwen3 MoE layer."""
    stats: list[PairwiseScalarStats] = []
    for module in model.modules():
        if not isinstance(module, moe_class):
            continue
        device = next(module.parameters()).device
        module.exfold_stats = PairwiseScalarStats(
            module.num_experts,
            ridge=ridge,
            clip_min=clip_min,
            clip_max=clip_max,
        ).to(device)
        stats.append(module.exfold_stats)
    if not stats:
        raise RuntimeError("no Qwen3 MoE layers were found")
    return stats
