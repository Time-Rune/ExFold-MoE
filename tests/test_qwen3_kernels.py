#!/usr/bin/env python3
"""Check ExFold Triton kernels against PyTorch reference code."""

from __future__ import annotations

import torch

from exfold.qwen3.kernels import (
    project_prefill_topk,
    remap_decode_static,
    reroute_decode_subset,
    select_and_reroute_decode,
)


LARGE_LOSS = 1e30


def select_target(
    source_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    projection_loss: torch.Tensor,
) -> torch.Tensor:
    """Select the single target with minimum projection loss."""
    candidate_loss = projection_loss[source_ids.unsqueeze(1), candidate_ids]
    return candidate_loss.argmin(dim=1)


def test_prefill() -> None:
    """Check that each removed slot is folded into one retained expert."""
    experts, top_k, tokens = 128, 8, 257
    coefficient = (torch.randn(experts, experts, device="cuda") * 0.1).contiguous()
    projection_loss = torch.rand(experts, experts, device="cuda").contiguous()
    projection_loss.diagonal().fill_(LARGE_LOSS)
    scores = torch.softmax(torch.randn(tokens, experts, device="cuda"), dim=-1)
    weights, expert_ids = scores.topk(top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    expert_ids = expert_ids.to(torch.int32)
    reference_ids = expert_ids[:, :4].long()
    reference_weights = weights[:, :4].clone()
    for position in range(4, top_k):
        source_ids = expert_ids[:, position].long()
        target_position = select_target(source_ids, reference_ids, projection_loss)
        target = reference_ids.gather(1, target_position.unsqueeze(1)).squeeze(1)
        reference_weights.scatter_add_(
            1,
            target_position.unsqueeze(1),
            weights[:, position : position + 1] * coefficient[source_ids, target].unsqueeze(1),
        )
    output_weights, output_ids = project_prefill_topk(
        weights,
        expert_ids,
        coefficient,
        4,
        projection_loss,
    )
    torch.cuda.synchronize()
    assert torch.equal(output_ids.long(), reference_ids)
    assert torch.allclose(output_weights, reference_weights, atol=5e-5, rtol=5e-5)


def test_decode() -> None:
    """Check minimum-loss remapping to the retained batch-level subset."""
    experts, top_k, batch_size, subset_size = 128, 8, 23, 64
    coefficient = (torch.randn(experts, experts, device="cuda") * 0.1).contiguous()
    projection_loss = torch.rand(experts, experts, device="cuda").contiguous()
    projection_loss.diagonal().fill_(LARGE_LOSS)
    scores = torch.softmax(torch.randn(batch_size, experts, device="cuda"), dim=-1)
    weights, expert_ids = scores.topk(top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    expert_ids = expert_ids.to(torch.int32)
    subset_ids = torch.randperm(experts, device="cuda")[:subset_size].to(torch.int32)
    source_ids = expert_ids.long()
    subset = subset_ids.long().unsqueeze(0).expand(batch_size, -1)
    in_subset = (source_ids.unsqueeze(-1) == subset.unsqueeze(1)).any(dim=-1)
    reference_weights = weights.clone()
    reference_ids = source_ids.clone()
    for position in range(top_k):
        source = source_ids[:, position]
        target_position = select_target(source, subset, projection_loss)
        target = subset.gather(1, target_position.unsqueeze(1)).squeeze(1)
        reference_weights[:, position] = torch.where(
            in_subset[:, position], weights[:, position], weights[:, position] * coefficient[source, target]
        )
        reference_ids[:, position] = torch.where(in_subset[:, position], source, target)
    output_weights, output_ids = reroute_decode_subset(
        weights,
        expert_ids,
        subset_ids,
        coefficient,
        projection_loss,
    )
    torch.cuda.synchronize()
    assert torch.equal(output_ids.long(), reference_ids)
    assert torch.allclose(output_weights, reference_weights, atol=5e-5, rtol=5e-5)


def test_unobserved_pair_is_noop() -> None:
    """Check that an unobserved pair contributes no projected output."""
    coefficient = torch.ones((8, 8), device="cuda").contiguous()
    projection_loss = torch.full((8, 8), LARGE_LOSS, device="cuda").contiguous()
    weights = torch.full((1, 8), 0.125, device="cuda")
    expert_ids = torch.arange(8, device="cuda", dtype=torch.int32).unsqueeze(0)
    prefill_weights, prefill_ids = project_prefill_topk(
        weights, expert_ids, coefficient, 4, projection_loss
    )
    decode_weights, decode_ids = reroute_decode_subset(
        weights, expert_ids, expert_ids[0, :4], coefficient, projection_loss
    )
    torch.cuda.synchronize()
    assert torch.allclose(prefill_weights, weights[:, :4])
    assert torch.equal(prefill_ids, expert_ids[:, :4])
    assert torch.allclose(decode_weights[:, :4], weights[:, :4])
    assert torch.equal(decode_weights[:, 4:], torch.zeros_like(decode_weights[:, 4:]))
    subset = expert_ids[0, :4]
    assert (decode_ids[:, 4:].unsqueeze(-1) == subset).any(dim=-1).all()


def test_decode_selection() -> None:
    """Check fused importance selection plus minimum-loss remapping."""
    experts, top_k, batch_size, subset_size = 128, 8, 32, 32
    coefficient = (torch.randn(experts, experts, device="cuda") * 0.1).contiguous()
    projection_loss = torch.rand(experts, experts, device="cuda").contiguous()
    projection_loss.diagonal().fill_(LARGE_LOSS)
    expert_norms = torch.rand(experts, device="cuda").add_(0.1).contiguous()
    scores = torch.softmax(torch.randn(batch_size, experts, device="cuda"), dim=-1)
    weights, expert_ids = scores.topk(top_k, dim=-1)
    expert_ids = expert_ids.to(torch.int32)

    flat_ids = expert_ids.reshape(-1).long()
    importance = torch.zeros(experts, device="cuda")
    importance.scatter_add_(
        0, flat_ids, weights.reshape(-1).float() * expert_norms[flat_ids]
    )
    subset_ids = importance.topk(subset_size, sorted=False).indices.to(torch.int32)
    reference_weights, reference_ids = reroute_decode_subset(
        weights, expert_ids, subset_ids, coefficient, projection_loss
    )
    output_weights, output_ids = select_and_reroute_decode(
        weights,
        expert_ids,
        expert_norms,
        coefficient,
        projection_loss,
        subset_size,
    )
    torch.cuda.synchronize()
    assert torch.equal(output_ids, reference_ids)
    assert torch.allclose(output_weights, reference_weights, atol=5e-5, rtol=5e-5)


def test_static_decode_remap() -> None:
    """Check the CUDA-Graph-friendly precomputed decode map."""
    experts, top_k, batch_size = 128, 8, 37
    weights = torch.rand(batch_size, top_k, device="cuda")
    expert_ids = torch.randint(
        experts, (batch_size, top_k), dtype=torch.int32, device="cuda"
    )
    target_ids = torch.arange(experts, dtype=torch.int32, device="cuda")
    target_ids[64:] -= 64
    target_scale = torch.linspace(0.5, 1.5, experts, device="cuda")
    reference_ids = target_ids[expert_ids.long()]
    reference_weights = weights * target_scale[expert_ids.long()]
    output_weights, output_ids = remap_decode_static(
        weights.clone(), expert_ids.clone(), target_ids, target_scale
    )
    torch.cuda.synchronize()
    assert torch.equal(output_ids, reference_ids)
    assert torch.allclose(output_weights, reference_weights, atol=5e-5, rtol=5e-5)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this test")
    torch.manual_seed(0)
    test_prefill()
    test_decode()
    test_unobserved_pair_is_noop()
    test_decode_selection()
    test_static_decode_remap()
    print("PASS")
