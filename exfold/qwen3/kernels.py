"""Triton kernels for ExFold prefill folding and decode remapping."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _project_prefill_kernel(
    weights_ptr,
    ids_ptr,
    coefficient_ptr,
    projection_loss_ptr,
    output_weights_ptr,
    output_ids_ptr,
    tokens,
    stride_weights_t,
    stride_weights_k,
    stride_ids_t,
    stride_ids_k,
    stride_coefficient_src,
    stride_coefficient_dst,
    stride_output_weights_t,
    stride_output_weights_k,
    stride_output_ids_t,
    stride_output_ids_k,
    K: tl.constexpr,
    TARGET_K: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    program_id = tl.program_id(0)
    token = program_id * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    valid_token = token < tokens
    target_position = tl.arange(0, TARGET_K)

    weight_offset = token[:, None] * stride_weights_t + target_position[None, :] * stride_weights_k
    id_offset = token[:, None] * stride_ids_t + target_position[None, :] * stride_ids_k
    output_weights = tl.load(weights_ptr + weight_offset, mask=valid_token[:, None], other=0.0).to(tl.float32)
    output_ids = tl.load(ids_ptr + id_offset, mask=valid_token[:, None], other=0).to(tl.int64)

    for source_position in tl.static_range(TARGET_K, K):
        source_weight = tl.load(
            weights_ptr + token * stride_weights_t + source_position * stride_weights_k,
            mask=valid_token,
            other=0.0,
        ).to(tl.float32)
        source_id = tl.load(
            ids_ptr + token * stride_ids_t + source_position * stride_ids_k,
            mask=valid_token,
            other=0,
        ).to(tl.int64)
        candidate_loss = tl.load(
            projection_loss_ptr
            + source_id[:, None] * stride_coefficient_src
            + output_ids * stride_coefficient_dst,
            mask=valid_token[:, None],
            other=1.0e30,
        )
        selected_position = tl.argmin(candidate_loss, axis=1)
        selected_loss = tl.min(candidate_loss, axis=1)
        target_id = tl.load(
            ids_ptr + token * stride_ids_t + selected_position * stride_ids_k,
            mask=valid_token,
            other=0,
        ).to(tl.int64)
        scale = tl.load(
            coefficient_ptr
            + source_id * stride_coefficient_src
            + target_id * stride_coefficient_dst,
            mask=valid_token,
            other=0.0,
        )
        scale = tl.where(selected_loss < 1.0e30, scale, 0.0)
        output_weights += tl.where(
            target_position[None, :] == selected_position[:, None],
            source_weight[:, None] * scale[:, None],
            0.0,
        )

    tl.store(
        output_weights_ptr
        + token[:, None] * stride_output_weights_t
        + target_position[None, :] * stride_output_weights_k,
        output_weights,
        mask=valid_token[:, None],
    )
    tl.store(
        output_ids_ptr
        + token[:, None] * stride_output_ids_t
        + target_position[None, :] * stride_output_ids_k,
        output_ids,
        mask=valid_token[:, None],
    )


def project_prefill_topk(
    weights: torch.Tensor,
    expert_ids: torch.Tensor,
    coefficient: torch.Tensor,
    target_k: int,
    projection_loss: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold each removed slot into the retained expert with minimum loss."""
    if weights.ndim != 2 or expert_ids.shape != weights.shape:
        raise ValueError("weights and expert_ids must be equally shaped matrices")
    if coefficient.ndim != 2 or coefficient.shape[0] != coefficient.shape[1]:
        raise ValueError("coefficient must be square")
    if projection_loss.shape != coefficient.shape:
        raise ValueError("projection_loss and coefficient must have the same shape")
    if not (weights.is_cuda and expert_ids.is_cuda and coefficient.is_cuda):
        raise RuntimeError("project_prefill_topk requires CUDA tensors")
    if not projection_loss.is_cuda:
        raise RuntimeError("projection_loss must be on CUDA")
    tokens, source_k = weights.shape
    target_k = max(1, min(int(target_k), source_k))
    if target_k == source_k:
        return weights.contiguous(), expert_ids.contiguous()
    output_weights = torch.empty((tokens, target_k), device=weights.device, dtype=weights.dtype)
    output_ids = torch.empty((tokens, target_k), device=expert_ids.device, dtype=expert_ids.dtype)
    if tokens == 0:
        return output_weights, output_ids
    weights = weights.contiguous()
    expert_ids = expert_ids.contiguous()
    coefficient = coefficient.contiguous()
    projection_loss = projection_loss.contiguous()
    block_tokens = 128 if tokens >= 128 else triton.next_power_of_2(tokens)
    _project_prefill_kernel[(triton.cdiv(tokens, block_tokens),)](
        weights,
        expert_ids,
        coefficient,
        projection_loss,
        output_weights,
        output_ids,
        tokens,
        weights.stride(0),
        weights.stride(1),
        expert_ids.stride(0),
        expert_ids.stride(1),
        coefficient.stride(0),
        coefficient.stride(1),
        output_weights.stride(0),
        output_weights.stride(1),
        output_ids.stride(0),
        output_ids.stride(1),
        K=source_k,
        TARGET_K=target_k,
        BLOCK_TOKENS=block_tokens,
        num_warps=4,
    )
    return output_weights, output_ids


@triton.jit
def _reroute_decode_kernel(
    weights_ptr,
    ids_ptr,
    subset_ids_ptr,
    coefficient_ptr,
    projection_loss_ptr,
    output_weights_ptr,
    output_ids_ptr,
    batch_size,
    subset_size,
    stride_weights_b,
    stride_weights_k,
    stride_ids_b,
    stride_ids_k,
    stride_coefficient_src,
    stride_coefficient_dst,
    stride_output_weights_b,
    stride_output_weights_k,
    stride_output_ids_b,
    stride_output_ids_k,
    K: tl.constexpr,
    BLOCK_SUBSET: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    program_id = tl.program_id(0)
    batch = program_id * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    valid_batch = batch < batch_size
    subset_position = tl.arange(0, BLOCK_SUBSET)
    valid_subset = subset_position < subset_size
    subset_ids = tl.load(subset_ids_ptr + subset_position, mask=valid_subset, other=0).to(tl.int64)

    for position in tl.static_range(K):
        source_weight = tl.load(
            weights_ptr + batch * stride_weights_b + position * stride_weights_k,
            mask=valid_batch,
            other=0.0,
        )
        source_id = tl.load(
            ids_ptr + batch * stride_ids_b + position * stride_ids_k,
            mask=valid_batch,
            other=0,
        ).to(tl.int64)
        in_subset = tl.sum(
            ((source_id[:, None] == subset_ids[None, :]) & valid_subset[None, :]).to(tl.int32),
            axis=1,
        ) > 0
        candidate_loss = tl.load(
            projection_loss_ptr
            + source_id[:, None] * stride_coefficient_src
            + subset_ids[None, :] * stride_coefficient_dst,
            mask=valid_batch[:, None] & valid_subset[None, :],
            other=3.0e30,
        )
        target_position = tl.argmin(candidate_loss, axis=1)
        selected_loss = tl.min(candidate_loss, axis=1)
        target_id = tl.load(subset_ids_ptr + target_position, mask=valid_batch, other=0).to(tl.int64)
        scale = tl.load(
            coefficient_ptr
            + source_id * stride_coefficient_src
            + target_id * stride_coefficient_dst,
            mask=valid_batch,
            other=0.0,
        ).to(source_weight.dtype)
        scale = tl.where(selected_loss < 1.0e30, scale, 0.0)
        tl.store(
            output_weights_ptr + batch * stride_output_weights_b + position * stride_output_weights_k,
            tl.where(
                in_subset,
                source_weight,
                source_weight * scale,
            ),
            mask=valid_batch,
        )
        tl.store(
            output_ids_ptr + batch * stride_output_ids_b + position * stride_output_ids_k,
            tl.where(in_subset, source_id, target_id),
            mask=valid_batch,
        )


def reroute_decode_subset(
    weights: torch.Tensor,
    expert_ids: torch.Tensor,
    subset_ids: torch.Tensor,
    coefficient: torch.Tensor,
    projection_loss: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remap decode routes outside the batch subset to minimum-loss targets."""
    if weights.ndim != 2 or expert_ids.shape != weights.shape:
        raise ValueError("weights and expert_ids must be equally shaped matrices")
    if subset_ids.ndim != 1 or subset_ids.numel() == 0:
        raise ValueError("subset_ids must be a non-empty vector")
    if coefficient.ndim != 2 or coefficient.shape[0] != coefficient.shape[1]:
        raise ValueError("coefficient must be square")
    if projection_loss.shape != coefficient.shape:
        raise ValueError("projection_loss and coefficient must have the same shape")
    if not (weights.is_cuda and expert_ids.is_cuda and subset_ids.is_cuda and coefficient.is_cuda):
        raise RuntimeError("reroute_decode_subset requires CUDA tensors")
    if not projection_loss.is_cuda:
        raise RuntimeError("projection_loss must be on CUDA")
    batch_size, top_k = weights.shape
    output_weights = torch.empty_like(weights)
    output_ids = torch.empty_like(expert_ids)
    if batch_size == 0:
        return output_weights, output_ids
    weights = weights.contiguous()
    expert_ids = expert_ids.contiguous()
    subset_ids = subset_ids.contiguous()
    coefficient = coefficient.contiguous()
    projection_loss = projection_loss.contiguous()
    block_subset = max(16, triton.next_power_of_2(subset_ids.numel()))
    block_batch = triton.next_power_of_2(min(batch_size, 64))
    _reroute_decode_kernel[(triton.cdiv(batch_size, block_batch),)](
        weights,
        expert_ids,
        subset_ids,
        coefficient,
        projection_loss,
        output_weights,
        output_ids,
        batch_size,
        subset_ids.numel(),
        weights.stride(0),
        weights.stride(1),
        expert_ids.stride(0),
        expert_ids.stride(1),
        coefficient.stride(0),
        coefficient.stride(1),
        output_weights.stride(0),
        output_weights.stride(1),
        output_ids.stride(0),
        output_ids.stride(1),
        K=top_k,
        BLOCK_SUBSET=block_subset,
        BLOCK_BATCH=block_batch,
        num_warps=4,
    )
    return output_weights, output_ids


@triton.jit
def _static_decode_remap_kernel(
    weights_ptr,
    ids_ptr,
    target_ids_ptr,
    target_scale_ptr,
    batch_size,
    stride_weights_b,
    stride_weights_k,
    stride_ids_b,
    stride_ids_k,
    K: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    batch = tl.program_id(0) * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    valid = batch < batch_size
    for position in tl.static_range(K):
        weight_offset = batch * stride_weights_b + position * stride_weights_k
        id_offset = batch * stride_ids_b + position * stride_ids_k
        source_weight = tl.load(weights_ptr + weight_offset, mask=valid, other=0.0)
        source_id = tl.load(ids_ptr + id_offset, mask=valid, other=0).to(tl.int64)
        target_id = tl.load(target_ids_ptr + source_id, mask=valid, other=0)
        scale = tl.load(target_scale_ptr + source_id, mask=valid, other=1.0)
        tl.store(weights_ptr + weight_offset, source_weight * scale, mask=valid)
        tl.store(ids_ptr + id_offset, target_id, mask=valid)


def remap_decode_static(
    weights: torch.Tensor,
    expert_ids: torch.Tensor,
    target_ids: torch.Tensor,
    target_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the precomputed paper-speed source-to-target map in place."""
    if weights.ndim != 2 or expert_ids.shape != weights.shape:
        raise ValueError("weights and expert_ids must be equally shaped matrices")
    if target_ids.ndim != 1 or target_scale.shape != target_ids.shape:
        raise ValueError("target_ids and target_scale must be equally shaped vectors")
    if not all(item.is_cuda for item in (weights, expert_ids, target_ids, target_scale)):
        raise RuntimeError("remap_decode_static requires CUDA tensors")
    if not weights.is_contiguous() or not expert_ids.is_contiguous():
        weights = weights.contiguous()
        expert_ids = expert_ids.contiguous()
    if weights.shape[0] == 0:
        return weights, expert_ids

    block_batch = max(1, triton.next_power_of_2(min(weights.shape[0], 128)))
    _static_decode_remap_kernel[(triton.cdiv(weights.shape[0], block_batch),)](
        weights,
        expert_ids,
        target_ids.contiguous(),
        target_scale.contiguous(),
        weights.shape[0],
        weights.stride(0),
        weights.stride(1),
        expert_ids.stride(0),
        expert_ids.stride(1),
        K=weights.shape[1],
        BLOCK_BATCH=block_batch,
        num_warps=4,
    )
    return weights, expert_ids


@triton.jit
def _zero_importance_kernel(
    importance_ptr,
    num_experts,
    BLOCK_EXPERTS: tl.constexpr,
):
    expert = tl.program_id(0) * BLOCK_EXPERTS + tl.arange(0, BLOCK_EXPERTS)
    tl.store(importance_ptr + expert, 0.0, mask=expert < num_experts)


@triton.jit
def _accumulate_decode_importance_kernel(
    weights_ptr,
    ids_ptr,
    norms_ptr,
    importance_ptr,
    routes,
    stride_weights_b,
    stride_weights_k,
    stride_ids_b,
    stride_ids_k,
    K: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
):
    route = tl.program_id(0) * BLOCK_ROUTES + tl.arange(0, BLOCK_ROUTES)
    valid = route < routes
    batch = route // K
    position = route % K
    expert_id = tl.load(
        ids_ptr + batch * stride_ids_b + position * stride_ids_k,
        mask=valid,
        other=0,
    ).to(tl.int64)
    weight = tl.load(
        weights_ptr + batch * stride_weights_b + position * stride_weights_k,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    norm = tl.load(norms_ptr + expert_id, mask=valid, other=0.0).to(tl.float32)
    tl.atomic_add(importance_ptr + expert_id, weight * norm, mask=valid)


@triton.jit
def _select_decode_subset_kernel(
    importance_ptr,
    subset_ids_ptr,
    num_experts,
    SUBSET_SIZE: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
):
    expert = tl.arange(0, BLOCK_EXPERTS)
    valid = expert < num_experts
    importance = tl.load(importance_ptr + expert, mask=valid, other=-float("inf"))

    # Positive float32 bit patterns preserve numeric ordering. Pack the expert
    # id into the low bits so Triton's value-only top-k also returns its index.
    importance_bits = importance.to(tl.uint32, bitcast=True).to(tl.uint64)
    tie_break = 0xFFFFFFFF - expert.to(tl.uint64)
    packed = (importance_bits << 32) | tie_break
    selected = tl.topk(packed, SUBSET_SIZE)
    selected_ids = 0xFFFFFFFF - (selected & 0xFFFFFFFF)
    tl.store(subset_ids_ptr + tl.arange(0, SUBSET_SIZE), selected_ids)


def select_and_reroute_decode(
    weights: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_norms: torch.Tensor,
    coefficient: torch.Tensor,
    projection_loss: torch.Tensor,
    subset_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the weighted Top-D expert pool and remap routes using Triton."""
    if weights.ndim != 2 or expert_ids.shape != weights.shape:
        raise ValueError("weights and expert_ids must be equally shaped matrices")
    if expert_norms.ndim != 1:
        raise ValueError("expert_norms must be a vector")
    num_experts = expert_norms.numel()
    if coefficient.shape != (num_experts, num_experts):
        raise ValueError("coefficient shape does not match expert_norms")
    if projection_loss.shape != coefficient.shape:
        raise ValueError("projection_loss and coefficient must have the same shape")
    if not all(
        item.is_cuda
        for item in (weights, expert_ids, expert_norms, coefficient, projection_loss)
    ):
        raise RuntimeError("select_and_reroute_decode requires CUDA tensors")

    batch_size, top_k = weights.shape
    subset_size = max(1, min(int(subset_size), num_experts))
    if subset_size == num_experts or batch_size == 0:
        return weights.contiguous(), expert_ids.contiguous()

    weights = weights.contiguous()
    expert_ids = expert_ids.contiguous()
    expert_norms = expert_norms.contiguous()
    importance = torch.empty(num_experts, device=weights.device, dtype=torch.float32)
    subset_ids = torch.empty(subset_size, device=expert_ids.device, dtype=torch.int32)
    routes = batch_size * top_k
    block_routes = 256
    block_experts = triton.next_power_of_2(num_experts)

    _zero_importance_kernel[(triton.cdiv(num_experts, 128),)](
        importance,
        num_experts,
        BLOCK_EXPERTS=128,
        num_warps=4,
    )
    _accumulate_decode_importance_kernel[(triton.cdiv(routes, block_routes),)](
        weights,
        expert_ids,
        expert_norms,
        importance,
        routes,
        weights.stride(0),
        weights.stride(1),
        expert_ids.stride(0),
        expert_ids.stride(1),
        K=top_k,
        BLOCK_ROUTES=block_routes,
        num_warps=4,
    )
    _select_decode_subset_kernel[(1,)](
        importance,
        subset_ids,
        num_experts,
        SUBSET_SIZE=subset_size,
        BLOCK_EXPERTS=block_experts,
        num_warps=4,
    )
    return reroute_decode_subset(
        weights, expert_ids, subset_ids, coefficient, projection_loss
    )
