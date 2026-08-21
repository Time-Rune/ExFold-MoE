"""Training-free pairwise scalar-projector calibration."""

from __future__ import annotations

import torch
from torch import nn


class PairwiseScalarStats(nn.Module):
    """Accumulate one MoE layer's directed scalar projectors."""

    default_loss = 1e30

    def __init__(
        self,
        num_experts: int,
        ridge: float = 1e-3,
        clip_min: float = -4.0,
        clip_max: float = 4.0,
    ) -> None:
        super().__init__()
        if num_experts <= 1:
            raise ValueError("num_experts must be greater than one")
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        if clip_min >= clip_max:
            raise ValueError("clip_min must be smaller than clip_max")

        shape = (num_experts, num_experts)
        self.register_buffer("dot", torch.zeros(shape, dtype=torch.float64), persistent=False)
        self.register_buffer(
            "target_sq_norm", torch.zeros(shape, dtype=torch.float64), persistent=False
        )
        self.register_buffer(
            "source_sq_norm", torch.zeros(shape, dtype=torch.float64), persistent=False
        )
        self.register_buffer(
            "expert_norm_sum", torch.zeros(num_experts, dtype=torch.float64), persistent=False
        )
        self.register_buffer(
            "expert_norm_count", torch.zeros(num_experts, dtype=torch.float64), persistent=False
        )
        self.num_experts = num_experts
        self.ridge = ridge
        self.clip_min = clip_min
        self.clip_max = clip_max

    @torch.no_grad()
    def update(self, expert_ids: torch.Tensor, expert_outputs: torch.Tensor) -> None:
        """Accumulate co-routed expert outputs.

        Args:
            expert_ids: Integer tensor with shape ``[tokens, top_k]``.
            expert_outputs: Unweighted expert outputs with shape
                ``[tokens, top_k, hidden_size]``.
        """
        if expert_ids.ndim != 2:
            raise ValueError("expert_ids must have shape [tokens, top_k]")
        if expert_outputs.shape[:2] != expert_ids.shape or expert_outputs.ndim != 3:
            raise ValueError(
                "expert_outputs must have shape [tokens, top_k, hidden_size]"
            )

        outputs = expert_outputs.float()
        pair_dot = torch.einsum("tsh,tdh->tsd", outputs, outputs)
        target_sq = outputs.square().sum(dim=-1).unsqueeze(1)
        source_sq = outputs.square().sum(dim=-1).unsqueeze(2)

        # The paper weights every source sample by ||u_i||_2.
        source_weight = source_sq.sqrt()
        source_ids = expert_ids.unsqueeze(2).expand_as(pair_dot)
        target_ids = expert_ids.unsqueeze(1).expand_as(pair_dot)
        pair_mask = source_ids != target_ids
        pair_index = (source_ids * self.num_experts + target_ids)[pair_mask]

        self.dot.view(-1).index_add_(
            0, pair_index, (pair_dot * source_weight)[pair_mask].double()
        )
        self.target_sq_norm.view(-1).index_add_(
            0, pair_index, (target_sq * source_weight)[pair_mask].double()
        )
        self.source_sq_norm.view(-1).index_add_(
            0,
            pair_index,
            (source_sq * source_weight).expand_as(pair_dot)[pair_mask].double(),
        )

        output_norms = outputs.norm(dim=-1)
        flat_ids = expert_ids.reshape(-1)
        self.expert_norm_sum.index_add_(0, flat_ids, output_norms.reshape(-1).double())
        self.expert_norm_count.index_add_(
            0,
            flat_ids,
            torch.ones_like(output_norms.reshape(-1), dtype=torch.float64),
        )

    @torch.no_grad()
    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the clipped scalar matrix and relative projection loss."""
        values = (self.dot, self.target_sq_norm, self.source_sq_norm)
        if not all(torch.isfinite(value).all() for value in values):
            raise RuntimeError("non-finite calibration statistics")

        observed = self.source_sq_norm > 0
        denominator = self.target_sq_norm + self.ridge
        coefficient = torch.zeros_like(self.dot)
        usable = observed & (denominator > 0)
        coefficient[usable] = self.dot[usable] / denominator[usable]
        coefficient.clamp_(self.clip_min, self.clip_max)

        residual = (
            self.source_sq_norm
            - 2 * coefficient * self.dot
            + coefficient.square() * self.target_sq_norm
        )
        loss = torch.full_like(self.source_sq_norm, self.default_loss)
        loss[observed] = residual[observed].clamp_min(0) / self.source_sq_norm[observed]
        loss.clamp_max_(self.default_loss)
        loss.fill_diagonal_(self.default_loss)

        coefficient.masked_fill_(~observed, 0.0)
        coefficient.masked_fill_(loss >= self.default_loss, 0.0)
        coefficient.fill_diagonal_(1.0)
        return coefficient.float(), loss.float()

    @torch.no_grad()
    def expert_norms(self) -> torch.Tensor:
        """Return the mean output norm of each routed expert."""
        count = self.expert_norm_count
        norms = self.expert_norm_sum / count.clamp_min(1)
        observed = count > 0
        fallback = norms[observed].median() if observed.any() else norms.new_tensor(1.0)
        return torch.where(observed, norms, fallback).float()
