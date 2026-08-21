#!/usr/bin/env python3
"""Check DeepSeek-V4 kernels against PyTorch and CUDA Graph replay."""

from __future__ import annotations

import torch

from exfold.deepseek_v4.cuda_ops import fold_decode, fold_prefill


def prefill_reference(weights, ids, coefficient, selection_loss, confidence_loss):
    order = weights.argsort(dim=1, descending=True)
    weights = weights.gather(1, order).float()
    ids = ids.gather(1, order).long()
    output = weights[:, :3].clone()
    retained = ids[:, :3]
    retained_weights = output.clone()
    retained_sum = retained_weights.sum(dim=1)
    fallback = torch.zeros_like(retained_sum)
    for position in range(3, weights.shape[1]):
        source = ids[:, position]
        losses = selection_loss[source.unsqueeze(1), retained]
        target_position = losses.argmin(dim=1)
        target = retained.gather(1, target_position[:, None]).squeeze(1)
        confidence = (1.0 - confidence_loss[source, target]).clamp(0.0, 1.0)
        transfer = weights[:, position] * confidence
        output.scatter_add_(
            1,
            target_position[:, None],
            (transfer * coefficient[source, target])[:, None],
        )
        fallback += weights[:, position] * (1.0 - confidence)
    output += retained_weights * (fallback / retained_sum)[:, None]
    return output, retained


def decode_reference(weights, ids, coefficient, loss, norms, budget):
    experts = coefficient.shape[0]
    importance = torch.zeros(experts, device=weights.device)
    flat_ids = ids.long().flatten()
    importance.scatter_add_(
        0, flat_ids, weights.float().flatten() * norms[flat_ids]
    )
    retained = importance.topk(budget, sorted=True).indices
    retained_mask = torch.zeros(experts, dtype=torch.bool, device=weights.device)
    retained_mask[retained] = True
    source = ids.long()
    candidate_loss = loss[source.unsqueeze(-1), retained]
    target = retained[candidate_loss.argmin(dim=-1)]
    target = torch.where(retained_mask[source], source, target)
    scale = torch.where(
        target == source,
        torch.ones_like(weights, dtype=torch.float32),
        coefficient[source, target],
    )
    return weights.float() * scale, target


def test_correctness() -> None:
    torch.manual_seed(20260809)
    experts, tokens, topk = 256, 64, 8
    weights = torch.rand(tokens, topk, device="cuda", dtype=torch.bfloat16)
    ids = torch.stack(
        [torch.randperm(experts, device="cuda")[:topk] for _ in range(tokens)]
    ).to(torch.int32)
    coefficient = torch.rand(experts, experts, device="cuda")
    selection_loss = torch.rand(experts, experts, device="cuda")
    confidence_loss = torch.rand(experts, experts, device="cuda")
    norms = torch.rand(experts, device="cuda").add_(0.1)

    expected_weights, expected_ids = prefill_reference(
        weights, ids, coefficient, selection_loss, confidence_loss
    )
    actual_weights, actual_ids = fold_prefill(
        weights, ids, coefficient, selection_loss, confidence_loss, 3, 2
    )
    torch.cuda.synchronize()
    assert torch.equal(actual_ids.long(), expected_ids)
    assert torch.allclose(
        actual_weights.float(), expected_weights, atol=2e-2, rtol=2e-2
    )

    expected_weights, expected_ids = decode_reference(
        weights, ids, coefficient, selection_loss, norms, 64
    )
    actual_weights, actual_ids = fold_decode(
        weights, ids, coefficient, selection_loss, norms, 64
    )
    torch.cuda.synchronize()
    assert torch.equal(actual_ids.long(), expected_ids)
    assert torch.allclose(
        actual_weights.float(), expected_weights, atol=2e-2, rtol=2e-2
    )


def test_cuda_graph() -> None:
    experts, tokens, topk = 256, 32, 8
    weights = torch.rand(tokens, topk, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(
        experts, (tokens, topk), device="cuda", dtype=torch.int32
    )
    coefficient = torch.rand(experts, experts, device="cuda")
    loss = torch.rand(experts, experts, device="cuda")
    norms = torch.rand(experts, device="cuda").add_(0.1)
    for _ in range(3):
        fold_decode(weights, ids, coefficient, loss, norms, 64)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_weights, graph_ids = fold_decode(
            weights, ids, coefficient, loss, norms, 64
        )
    graph.replay()
    torch.cuda.synchronize()
    assert graph_weights.shape == weights.shape
    assert graph_ids.shape == ids.shape


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this test")
    test_correctness()
    test_cuda_graph()
    print("PASS")
