#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_radix_sort.cuh>
#include <torch/extension.h>

#include <cfloat>
#include <cmath>
#include <tuple>

namespace {

constexpr int kExperts = 256;
constexpr int kThreads = 256;
constexpr int kMaxTopK = 8;
constexpr int kMhcMult = 4;
constexpr int kMhcMixWidth = (2 + kMhcMult) * kMhcMult;
constexpr int kMhcThreads = 32;

__global__ void mhc_sinkhorn_kernel(
    const float* mixes,
    const float* hc_scale,
    const float* hc_base,
    float* pre,
    float* post,
    float* comb,
    int tokens,
    int sinkhorn_iters,
    float eps) {
  const int token = blockIdx.x;
  const int lane = threadIdx.x;
  if (token >= tokens) {
    return;
  }

  __shared__ float local_mix[kMhcMixWidth];
  __shared__ float local_comb[kMhcMult * kMhcMult];
  if (lane < kMhcMixWidth) {
    local_mix[lane] = mixes[token * kMhcMixWidth + lane];
  }
  __syncwarp();

  if (lane < kMhcMult) {
    pre[token * kMhcMult + lane] =
        1.0f / (1.0f + expf(-(
            local_mix[lane] * hc_scale[0] + hc_base[lane]))) + eps;
    post[token * kMhcMult + lane] =
        2.0f / (1.0f + expf(-(
            local_mix[kMhcMult + lane] * hc_scale[1] +
            hc_base[kMhcMult + lane])));

    const int row_offset = lane * kMhcMult;
    float row_max = -FLT_MAX;
    #pragma unroll
    for (int col = 0; col < kMhcMult; ++col) {
      const int index = row_offset + col;
      const float value =
          local_mix[2 * kMhcMult + index] * hc_scale[2] +
          hc_base[2 * kMhcMult + index];
      local_comb[index] = value;
      row_max = fmaxf(row_max, value);
    }
    float row_sum = 0.0f;
    #pragma unroll
    for (int col = 0; col < kMhcMult; ++col) {
      const int index = row_offset + col;
      const float value = expf(local_comb[index] - row_max);
      local_comb[index] = value;
      row_sum += value;
    }
    #pragma unroll
    for (int col = 0; col < kMhcMult; ++col) {
      const int index = row_offset + col;
      local_comb[index] = local_comb[index] / row_sum + eps;
    }
  }
  __syncwarp();

  if (lane < kMhcMult) {
    float col_sum = 0.0f;
    #pragma unroll
    for (int row = 0; row < kMhcMult; ++row) {
      col_sum += local_comb[row * kMhcMult + lane];
    }
    #pragma unroll
    for (int row = 0; row < kMhcMult; ++row) {
      const int index = row * kMhcMult + lane;
      local_comb[index] /= col_sum + eps;
    }
  }
  __syncwarp();

  for (int iteration = 1; iteration < sinkhorn_iters; ++iteration) {
    if (lane < kMhcMult) {
      const int row_offset = lane * kMhcMult;
      float row_sum = 0.0f;
      #pragma unroll
      for (int col = 0; col < kMhcMult; ++col) {
        row_sum += local_comb[row_offset + col];
      }
      #pragma unroll
      for (int col = 0; col < kMhcMult; ++col) {
        local_comb[row_offset + col] /= row_sum + eps;
      }
    }
    __syncwarp();
    if (lane < kMhcMult) {
      float col_sum = 0.0f;
      #pragma unroll
      for (int row = 0; row < kMhcMult; ++row) {
        col_sum += local_comb[row * kMhcMult + lane];
      }
      #pragma unroll
      for (int row = 0; row < kMhcMult; ++row) {
        const int index = row * kMhcMult + lane;
        local_comb[index] /= col_sum + eps;
      }
    }
    __syncwarp();
  }

  if (lane < kMhcMult * kMhcMult) {
    comb[token * kMhcMult * kMhcMult + lane] = local_comb[lane];
  }
}

template <typename scalar_t, typename index_t>
__global__ void prefill_kernel(
    const scalar_t* weights,
    const index_t* ids,
    const float* coefficient,
    const float* selection_loss,
    const float* confidence_loss,
    scalar_t* output_weights,
    index_t* output_ids,
    int tokens,
    int topk,
    int target_k,
    int renorm_mode) {
  const int token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= tokens) {
    return;
  }

  float local_weights[kMaxTopK];
  float retained_weights[kMaxTopK];
  int local_ids[kMaxTopK];
  float original_l2_sq = 0.0f;
  #pragma unroll
  for (int position = 0; position < kMaxTopK; ++position) {
    if (position < topk) {
      local_weights[position] = static_cast<float>(weights[token * topk + position]);
      local_ids[position] = static_cast<int>(ids[token * topk + position]);
      original_l2_sq += local_weights[position] * local_weights[position];
    }
  }

  for (int position = 1; position < topk; ++position) {
    const float current_weight = local_weights[position];
    const int current_id = local_ids[position];
    int insert = position;
    while (insert > 0 && current_weight > local_weights[insert - 1]) {
      local_weights[insert] = local_weights[insert - 1];
      local_ids[insert] = local_ids[insert - 1];
      --insert;
    }
    local_weights[insert] = current_weight;
    local_ids[insert] = current_id;
  }

  float retained_sum = 0.0f;
  for (int position = 0; position < target_k; ++position) {
    retained_weights[position] = local_weights[position];
    retained_sum += retained_weights[position];
  }
  float fallback_weight = 0.0f;
  for (int source_position = target_k; source_position < topk; ++source_position) {
    const int source = local_ids[source_position];
    float best_loss = FLT_MAX;
    int best_position = 0;
    for (int target_position = 0; target_position < target_k; ++target_position) {
      const int target = local_ids[target_position];
      const float candidate = selection_loss[source * kExperts + target];
      if (candidate < best_loss) {
        best_loss = candidate;
        best_position = target_position;
      }
    }
    const int target = local_ids[best_position];
    float transfer_confidence = 1.0f;
    if (renorm_mode == 2) {
      const float selected_confidence_loss =
          confidence_loss[source * kExperts + target];
      transfer_confidence =
          fminf(fmaxf(1.0f - selected_confidence_loss, 0.0f), 1.0f);
      fallback_weight +=
          local_weights[source_position] * (1.0f - transfer_confidence);
    }
    local_weights[best_position] +=
        local_weights[source_position] * coefficient[source * kExperts + target] *
        transfer_confidence;
  }

  if (renorm_mode == 2 && retained_sum > 1.0e-20f) {
    const float fallback_scale = fallback_weight / retained_sum;
    for (int position = 0; position < target_k; ++position) {
      local_weights[position] += retained_weights[position] * fallback_scale;
    }
  }

  float output_scale = 1.0f;
  if (renorm_mode == 1) {
    float folded_l2_sq = 0.0f;
    for (int position = 0; position < target_k; ++position) {
      folded_l2_sq += local_weights[position] * local_weights[position];
    }
    if (folded_l2_sq > 1.0e-20f) {
      output_scale = sqrtf(original_l2_sq / folded_l2_sq);
    }
  }

  for (int position = 0; position < target_k; ++position) {
    output_weights[token * target_k + position] =
        static_cast<scalar_t>(local_weights[position] * output_scale);
    output_ids[token * target_k + position] =
        static_cast<index_t>(local_ids[position]);
  }
}

template <typename scalar_t, typename index_t>
__global__ void importance_protected_kernel(
    const scalar_t* weights,
    const index_t* ids,
    const float* norms,
    float* importance,
    int tokens,
    int topk,
    int protected_topk,
    int importance_mode) {
  const int token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= tokens) {
    return;
  }

  float best_weights[kMaxTopK];
  int best_ids[kMaxTopK];
  #pragma unroll
  for (int position = 0; position < kMaxTopK; ++position) {
    best_weights[position] = -FLT_MAX;
    best_ids[position] = 0;
  }

  for (int position = 0; position < topk; ++position) {
    const float candidate_weight =
        static_cast<float>(weights[token * topk + position]);
    const int candidate_id = static_cast<int>(ids[token * topk + position]);
    int insert = protected_topk;
    while (insert > 0 && candidate_weight > best_weights[insert - 1]) {
      if (insert < protected_topk) {
        best_weights[insert] = best_weights[insert - 1];
        best_ids[insert] = best_ids[insert - 1];
      }
      --insert;
    }
    if (insert < protected_topk) {
      best_weights[insert] = candidate_weight;
      best_ids[insert] = candidate_id;
    }
  }

  for (int position = 0; position < protected_topk; ++position) {
    const int expert = best_ids[position];
    if (importance_mode == 2) {
      atomicExch(importance + expert, norms[expert]);
    } else {
      const float score = best_weights[position];
      atomicAdd(
          importance + expert,
          importance_mode == 1 ? score : score * norms[expert]);
    }
  }
}

template <typename scalar_t, typename index_t>
__global__ void fused_importance_select_and_apply_kernel(
    const float* coefficient,
    const float* loss,
    const float* norms,
    const scalar_t* weights,
    const index_t* ids,
    scalar_t* output_weights,
    index_t* output_ids,
    int tokens,
    int topk,
    int protected_topk,
    int budget,
    int mode,
    int importance_mode) {
  using Sort = cub::BlockRadixSort<float, kThreads, 1, int>;
  __shared__ typename Sort::TempStorage sort_storage;
  __shared__ float importance[kExperts];
  __shared__ int retained_ids[kExperts];
  __shared__ float retained_importance[kExperts];
  __shared__ int retained_mask[kExperts];
  __shared__ int target_ids[kExperts];
  __shared__ float target_scales[kExperts];

  const int source = threadIdx.x;
  importance[source] = 0.0f;
  retained_mask[source] = 0;
  __syncthreads();

  if (mode == 0) {
    const int routes = tokens * topk;
    for (int route = source; route < routes; route += kThreads) {
      const int expert = static_cast<int>(ids[route]);
      if (importance_mode == 2) {
        atomicExch(importance + expert, norms[expert]);
      } else {
        const float score = static_cast<float>(weights[route]);
        atomicAdd(
            importance + expert,
            importance_mode == 1 ? score : score * norms[expert]);
      }
    }
  } else {
    for (int token = source; token < tokens; token += kThreads) {
      float best_weights[kMaxTopK];
      int best_ids[kMaxTopK];
      #pragma unroll
      for (int position = 0; position < kMaxTopK; ++position) {
        best_weights[position] = -FLT_MAX;
        best_ids[position] = 0;
      }

      for (int position = 0; position < topk; ++position) {
        const float candidate_weight =
            static_cast<float>(weights[token * topk + position]);
        const int candidate_id =
            static_cast<int>(ids[token * topk + position]);
        int insert = protected_topk;
        while (insert > 0 && candidate_weight > best_weights[insert - 1]) {
          if (insert < protected_topk) {
            best_weights[insert] = best_weights[insert - 1];
            best_ids[insert] = best_ids[insert - 1];
          }
          --insert;
        }
        if (insert < protected_topk) {
          best_weights[insert] = candidate_weight;
          best_ids[insert] = candidate_id;
        }
      }

      for (int position = 0; position < protected_topk; ++position) {
        const int expert = best_ids[position];
        if (importance_mode == 2) {
          atomicExch(importance + expert, norms[expert]);
        } else {
          const float score = best_weights[position];
          atomicAdd(
              importance + expert,
              importance_mode == 1 ? score : score * norms[expert]);
        }
      }
    }
  }
  __syncthreads();

  float keys[1] = {importance[source]};
  int experts[1] = {source};
  Sort(sort_storage).SortDescending(keys, experts);
  retained_ids[source] = experts[0];
  retained_importance[source] = keys[0];
  __syncthreads();

  if (source < budget && retained_importance[source] > 0.0f) {
    retained_mask[retained_ids[source]] = 1;
  }
  __syncthreads();

  if (retained_mask[source]) {
    target_ids[source] = source;
    target_scales[source] = 1.0f;
  } else if (mode == 0 && importance[source] <= 0.0f) {
    // In fixed-budget mode every routed expert contributes importance, so an
    // inactive expert's mapping is never consumed by the route rewrite.
    target_ids[source] = source;
    target_scales[source] = 1.0f;
  } else {
    float best_loss = FLT_MAX;
    int best_target = source;
    for (int position = 0; position < budget; ++position) {
      if (retained_importance[position] <= 0.0f) {
        continue;
      }
      const int target = retained_ids[position];
      const float candidate = loss[source * kExperts + target];
      if (candidate < best_loss) {
        best_loss = candidate;
        best_target = target;
      }
    }
    target_ids[source] = best_target;
    target_scales[source] = best_target == source
        ? 1.0f
        : coefficient[source * kExperts + best_target];
  }
  __syncthreads();

  const int routes = tokens * topk;
  for (int route = source; route < routes; route += kThreads) {
    const int route_source = static_cast<int>(ids[route]);
    output_ids[route] = static_cast<index_t>(target_ids[route_source]);
    output_weights[route] = static_cast<scalar_t>(
        static_cast<float>(weights[route]) * target_scales[route_source]);
  }
}

template <typename scalar_t, typename index_t>
__global__ void select_and_compact_kernel(
    const float* importance,
    const float* coefficient,
    const float* loss,
    const scalar_t* weights,
    const index_t* ids,
    scalar_t* output_weights,
    index_t* output_ids,
    int tokens,
    int topk,
    int protected_topk,
    int budget) {
  using Sort = cub::BlockRadixSort<float, kThreads, 1, int>;
  __shared__ typename Sort::TempStorage sort_storage;
  __shared__ int retained_ids[kExperts];
  __shared__ float retained_importance[kExperts];
  __shared__ int target_ids[kExperts];

  const int source_expert = threadIdx.x;
  float keys[1] = {importance[source_expert]};
  int experts[1] = {source_expert};
  Sort(sort_storage).SortDescending(keys, experts);
  retained_ids[source_expert] = experts[0];
  retained_importance[source_expert] = keys[0];
  __syncthreads();

  bool retained = false;
  for (int position = 0; position < budget; ++position) {
    retained = retained ||
        (retained_importance[position] > 0.0f &&
         retained_ids[position] == source_expert);
  }
  if (retained) {
    target_ids[source_expert] = source_expert;
  } else {
    float best_loss = FLT_MAX;
    int best_target = source_expert;
    for (int position = 0; position < budget; ++position) {
      if (retained_importance[position] <= 0.0f) {
        continue;
      }
      const int target = retained_ids[position];
      const float candidate = loss[source_expert * kExperts + target];
      if (candidate < best_loss) {
        best_loss = candidate;
        best_target = target;
      }
    }
    target_ids[source_expert] = best_target;
  }
  __syncthreads();

  for (int token = source_expert; token < tokens; token += kThreads) {
    float local_weights[kMaxTopK];
    int local_ids[kMaxTopK];
    float compact_weights[kMaxTopK];
    int compact_ids[kMaxTopK];

    #pragma unroll
    for (int position = 0; position < kMaxTopK; ++position) {
      local_weights[position] = -FLT_MAX;
      local_ids[position] = 0;
      compact_weights[position] = 0.0f;
      compact_ids[position] = 0;
      if (position < topk) {
        local_weights[position] =
            static_cast<float>(weights[token * topk + position]);
        local_ids[position] = static_cast<int>(ids[token * topk + position]);
      }
    }

    for (int position = 1; position < topk; ++position) {
      const float current_weight = local_weights[position];
      const int current_id = local_ids[position];
      int insert = position;
      while (insert > 0 && current_weight > local_weights[insert - 1]) {
        local_weights[insert] = local_weights[insert - 1];
        local_ids[insert] = local_ids[insert - 1];
        --insert;
      }
      local_weights[insert] = current_weight;
      local_ids[insert] = current_id;
    }

    for (int position = 0; position < protected_topk; ++position) {
      compact_ids[position] = target_ids[local_ids[position]];
    }

    for (int source_position = 0; source_position < topk; ++source_position) {
      const int source = local_ids[source_position];
      float best_loss = FLT_MAX;
      int best_position = 0;
      for (int target_position = 0; target_position < protected_topk;
           ++target_position) {
        const int target = compact_ids[target_position];
        const float candidate =
            source == target ? -1.0f : loss[source * kExperts + target];
        if (candidate < best_loss) {
          best_loss = candidate;
          best_position = target_position;
        }
      }
      const int target = compact_ids[best_position];
      const float scale =
          source == target ? 1.0f : coefficient[source * kExperts + target];
      compact_weights[best_position] += local_weights[source_position] * scale;
    }

    for (int position = 0; position < protected_topk; ++position) {
      output_weights[token * protected_topk + position] =
          static_cast<scalar_t>(compact_weights[position]);
      output_ids[token * protected_topk + position] =
          static_cast<index_t>(compact_ids[position]);
    }
  }
}

void check_common(
    const torch::Tensor& weights,
    const torch::Tensor& ids,
    const torch::Tensor& coefficient,
    const torch::Tensor& loss) {
  TORCH_CHECK(weights.is_cuda() && ids.is_cuda(), "routing tensors must be CUDA");
  TORCH_CHECK(
      coefficient.is_cuda() && loss.is_cuda(), "calibration tensors must be CUDA");
  TORCH_CHECK(weights.is_contiguous() && ids.is_contiguous(), "routing tensors must be contiguous");
  TORCH_CHECK(
      coefficient.is_contiguous() && loss.is_contiguous(),
      "calibration tensors must be contiguous");
  TORCH_CHECK(weights.dim() == 2 && ids.sizes() == weights.sizes(), "routing tensors must be [tokens, topk]");
  TORCH_CHECK(
      coefficient.scalar_type() == torch::kFloat && loss.scalar_type() == torch::kFloat,
      "calibration tensors must be float32");
  TORCH_CHECK(
      coefficient.sizes() == torch::IntArrayRef({kExperts, kExperts}) &&
          loss.sizes() == coefficient.sizes(),
      "DeepSeek-V4 calibration must be [256, 256]");
  TORCH_CHECK(weights.size(1) <= kMaxTopK, "topk exceeds the fused kernel limit");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mhc_sinkhorn_cuda(
    const torch::Tensor& mixes,
    const torch::Tensor& hc_scale,
    const torch::Tensor& hc_base,
    int64_t hc_mult,
    int64_t sinkhorn_iters,
    double eps) {
  TORCH_CHECK(
      mixes.is_cuda() && hc_scale.is_cuda() && hc_base.is_cuda(),
      "MHC tensors must be CUDA");
  TORCH_CHECK(
      mixes.is_contiguous() && hc_scale.is_contiguous() && hc_base.is_contiguous(),
      "MHC tensors must be contiguous");
  TORCH_CHECK(
      mixes.scalar_type() == torch::kFloat &&
          hc_scale.scalar_type() == torch::kFloat &&
          hc_base.scalar_type() == torch::kFloat,
      "MHC tensors must be float32");
  TORCH_CHECK(mixes.dim() == 3, "mixes must be [batch, sequence, 24]");
  TORCH_CHECK(hc_mult == kMhcMult, "only hc_mult=4 is supported");
  TORCH_CHECK(
      mixes.size(2) == kMhcMixWidth,
      "mixes last dimension must be 24");
  TORCH_CHECK(
      hc_scale.numel() == 3 && hc_base.numel() == kMhcMixWidth,
      "invalid MHC scale/base shape");
  TORCH_CHECK(sinkhorn_iters > 0, "sinkhorn_iters must be positive");

  const at::cuda::OptionalCUDAGuard guard(mixes.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto pre = torch::empty(
      {mixes.size(0), mixes.size(1), kMhcMult}, mixes.options());
  auto post = torch::empty_like(pre);
  auto comb = torch::empty(
      {mixes.size(0), mixes.size(1), kMhcMult, kMhcMult}, mixes.options());
  const int tokens = mixes.size(0) * mixes.size(1);
  if (tokens > 0) {
    mhc_sinkhorn_kernel<<<tokens, kMhcThreads, 0, stream>>>(
        mixes.data_ptr<float>(),
        hc_scale.data_ptr<float>(),
        hc_base.data_ptr<float>(),
        pre.data_ptr<float>(),
        post.data_ptr<float>(),
        comb.data_ptr<float>(),
        tokens,
        sinkhorn_iters,
        static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {pre, post, comb};
}

std::tuple<torch::Tensor, torch::Tensor> exfold_prefill_cuda(
    const torch::Tensor& weights,
    const torch::Tensor& ids,
    const torch::Tensor& coefficient,
    const torch::Tensor& selection_loss,
    const torch::Tensor& confidence_loss,
    int64_t target_k,
    int64_t renorm_mode) {
  check_common(weights, ids, coefficient, selection_loss);
  TORCH_CHECK(confidence_loss.is_cuda(), "confidence_loss must be CUDA");
  TORCH_CHECK(confidence_loss.is_contiguous(), "confidence_loss must be contiguous");
  TORCH_CHECK(
      confidence_loss.scalar_type() == torch::kFloat,
      "confidence_loss must be float32");
  TORCH_CHECK(
      confidence_loss.sizes() == coefficient.sizes(),
      "confidence_loss must match coefficient shape");
  TORCH_CHECK(target_k > 0 && target_k <= weights.size(1), "invalid target_k");
  TORCH_CHECK(
      renorm_mode >= 0 && renorm_mode <= 2,
      "prefill renorm_mode must be 0 (none), 1 (router_l2), or 2 "
      "(similarity_blend)");
  const at::cuda::OptionalCUDAGuard guard(weights.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto output_weights = torch::empty(
      {weights.size(0), target_k}, weights.options());
  auto output_ids = torch::empty({ids.size(0), target_k}, ids.options());
  const int blocks = (weights.size(0) + kThreads - 1) / kThreads;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, weights.scalar_type(), "exfold_prefill_weights", [&] {
        using weight_t = scalar_t;
        AT_DISPATCH_INTEGRAL_TYPES(ids.scalar_type(), "exfold_prefill_ids", [&] {
          using index_t = scalar_t;
          prefill_kernel<weight_t, index_t><<<blocks, kThreads, 0, stream>>>(
              weights.data_ptr<weight_t>(),
              ids.data_ptr<index_t>(),
              coefficient.data_ptr<float>(),
              selection_loss.data_ptr<float>(),
              confidence_loss.data_ptr<float>(),
              output_weights.data_ptr<weight_t>(),
              output_ids.data_ptr<index_t>(),
              weights.size(0),
              weights.size(1),
              target_k,
              renorm_mode);
        });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output_weights, output_ids};
}

std::tuple<torch::Tensor, torch::Tensor> exfold_decode_cuda(
    const torch::Tensor& weights,
    const torch::Tensor& ids,
    const torch::Tensor& coefficient,
    const torch::Tensor& loss,
    const torch::Tensor& norms,
    int64_t protected_topk,
    int64_t budget,
    int64_t mode,
    int64_t importance_mode) {
  check_common(weights, ids, coefficient, loss);
  TORCH_CHECK(norms.is_cuda() && norms.is_contiguous(), "norms must be contiguous CUDA");
  TORCH_CHECK(
      norms.scalar_type() == torch::kFloat && norms.numel() == kExperts,
      "norms must be float32 [256]");
  TORCH_CHECK(budget > 0 && budget <= kExperts, "invalid expert budget");
  TORCH_CHECK(mode == 0 || mode == 1, "mode must be budget or protected union");
  TORCH_CHECK(
      importance_mode >= 0 && importance_mode <= 2,
      "importance_mode must be score_norm, score, or norm");
  TORCH_CHECK(
      protected_topk > 0 && protected_topk <= weights.size(1),
      "invalid protected_topk");

  const at::cuda::OptionalCUDAGuard guard(weights.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto output_weights = torch::empty_like(weights);
  auto output_ids = torch::empty_like(ids);

  const int tokens = weights.size(0);
  const int topk = weights.size(1);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, weights.scalar_type(), "exfold_decode_weights", [&] {
        using weight_t = scalar_t;
        AT_DISPATCH_INTEGRAL_TYPES(ids.scalar_type(), "exfold_decode_ids", [&] {
          using index_t = scalar_t;
          fused_importance_select_and_apply_kernel<weight_t, index_t>
              <<<1, kThreads, 0, stream>>>(
              coefficient.data_ptr<float>(),
              loss.data_ptr<float>(),
              norms.data_ptr<float>(),
              weights.data_ptr<weight_t>(),
              ids.data_ptr<index_t>(),
              output_weights.data_ptr<weight_t>(),
              output_ids.data_ptr<index_t>(),
              tokens,
              topk,
              protected_topk,
              budget,
              mode,
              importance_mode);
        });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output_weights, output_ids};
}

std::tuple<torch::Tensor, torch::Tensor> exfold_decode_compact_cuda(
    const torch::Tensor& weights,
    const torch::Tensor& ids,
    const torch::Tensor& coefficient,
    const torch::Tensor& loss,
    const torch::Tensor& norms,
    int64_t protected_topk,
    int64_t budget,
    int64_t importance_mode) {
  check_common(weights, ids, coefficient, loss);
  TORCH_CHECK(norms.is_cuda() && norms.is_contiguous(), "norms must be contiguous CUDA");
  TORCH_CHECK(
      norms.scalar_type() == torch::kFloat && norms.numel() == kExperts,
      "norms must be float32 [256]");
  TORCH_CHECK(budget > 0 && budget <= kExperts, "invalid expert budget");
  TORCH_CHECK(
      importance_mode >= 0 && importance_mode <= 2,
      "importance_mode must be score_norm, score, or norm");
  TORCH_CHECK(
      protected_topk > 0 && protected_topk <= weights.size(1),
      "invalid protected_topk");

  const at::cuda::OptionalCUDAGuard guard(weights.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto importance = torch::empty({kExperts}, weights.options().dtype(torch::kFloat));
  auto output_weights = torch::empty(
      {weights.size(0), protected_topk}, weights.options());
  auto output_ids = torch::empty({ids.size(0), protected_topk}, ids.options());
  cudaMemsetAsync(importance.data_ptr(), 0, kExperts * sizeof(float), stream);

  const int tokens = weights.size(0);
  const int topk = weights.size(1);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, weights.scalar_type(), "exfold_decode_compact_weights", [&] {
        using weight_t = scalar_t;
        AT_DISPATCH_INTEGRAL_TYPES(ids.scalar_type(), "exfold_decode_compact_ids", [&] {
          using index_t = scalar_t;
          importance_protected_kernel<weight_t, index_t>
              <<<(tokens + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                  weights.data_ptr<weight_t>(),
                  ids.data_ptr<index_t>(),
                  norms.data_ptr<float>(),
                  importance.data_ptr<float>(),
                  tokens,
                  topk,
                  protected_topk,
                  importance_mode);
          select_and_compact_kernel<weight_t, index_t><<<1, kThreads, 0, stream>>>(
              importance.data_ptr<float>(),
              coefficient.data_ptr<float>(),
              loss.data_ptr<float>(),
              weights.data_ptr<weight_t>(),
              ids.data_ptr<index_t>(),
              output_weights.data_ptr<weight_t>(),
              output_ids.data_ptr<index_t>(),
              tokens,
              topk,
              protected_topk,
              budget);
        });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output_weights, output_ids};
}
