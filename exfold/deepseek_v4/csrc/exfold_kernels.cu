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
    int budget) {
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

  const int routes = tokens * topk;
  for (int route = source; route < routes; route += kThreads) {
    const int expert = static_cast<int>(ids[route]);
    const float score = static_cast<float>(weights[route]);
    atomicAdd(importance + expert, score * norms[expert]);
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
  } else if (importance[source] <= 0.0f) {
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

  for (int route = source; route < routes; route += kThreads) {
    const int route_source = static_cast<int>(ids[route]);
    output_ids[route] = static_cast<index_t>(target_ids[route_source]);
    output_weights[route] = static_cast<scalar_t>(
        static_cast<float>(weights[route]) * target_scales[route_source]);
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
    int64_t budget) {
  check_common(weights, ids, coefficient, loss);
  TORCH_CHECK(norms.is_cuda() && norms.is_contiguous(), "norms must be contiguous CUDA");
  TORCH_CHECK(
      norms.scalar_type() == torch::kFloat && norms.numel() == kExperts,
      "norms must be float32 [256]");
  TORCH_CHECK(budget > 0 && budget <= kExperts, "invalid expert budget");

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
              budget);
        });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output_weights, output_ids};
}
