#include <torch/extension.h>

#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> exfold_prefill_cuda(
    const torch::Tensor& weights,
    const torch::Tensor& ids,
    const torch::Tensor& coefficient,
    const torch::Tensor& selection_loss,
    const torch::Tensor& confidence_loss,
    int64_t target_k,
    int64_t renorm_mode);

std::tuple<torch::Tensor, torch::Tensor> exfold_decode_cuda(
    const torch::Tensor& weights,
    const torch::Tensor& ids,
    const torch::Tensor& coefficient,
    const torch::Tensor& loss,
    const torch::Tensor& norms,
    int64_t protected_topk,
    int64_t budget,
    int64_t mode,
    int64_t importance_mode);

std::tuple<torch::Tensor, torch::Tensor> exfold_decode_compact_cuda(
    const torch::Tensor& weights,
    const torch::Tensor& ids,
    const torch::Tensor& coefficient,
    const torch::Tensor& loss,
    const torch::Tensor& norms,
    int64_t protected_topk,
    int64_t budget,
    int64_t importance_mode);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mhc_sinkhorn_cuda(
    const torch::Tensor& mixes,
    const torch::Tensor& hc_scale,
    const torch::Tensor& hc_base,
    int64_t hc_mult,
    int64_t sinkhorn_iters,
    double eps);

TORCH_LIBRARY(dsv4_exfold, m) {
  m.def(
      "prefill(Tensor weights, Tensor ids, Tensor coefficient, "
      "Tensor selection_loss, Tensor confidence_loss, int target_k, "
      "int renorm_mode) -> (Tensor, Tensor)");
  m.def(
      "decode(Tensor weights, Tensor ids, Tensor coefficient, Tensor loss, "
      "Tensor norms, int protected_topk, int budget, int mode, "
      "int importance_mode) "
      "-> (Tensor, Tensor)");
  m.def(
      "decode_compact(Tensor weights, Tensor ids, Tensor coefficient, "
      "Tensor loss, Tensor norms, int protected_topk, int budget, "
      "int importance_mode) -> (Tensor, Tensor)");
  m.def(
      "mhc_sinkhorn(Tensor mixes, Tensor hc_scale, Tensor hc_base, "
      "int hc_mult, int sinkhorn_iters, float eps) -> "
      "(Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(dsv4_exfold, CUDA, m) {
  m.impl("prefill", &exfold_prefill_cuda);
  m.impl("decode", &exfold_decode_cuda);
  m.impl("decode_compact", &exfold_decode_compact_cuda);
  m.impl("mhc_sinkhorn", &mhc_sinkhorn_cuda);
}
