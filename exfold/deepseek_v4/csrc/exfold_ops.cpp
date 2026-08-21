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
    int64_t budget);

TORCH_LIBRARY(dsv4_exfold, m) {
  m.def(
      "prefill(Tensor weights, Tensor ids, Tensor coefficient, "
      "Tensor selection_loss, Tensor confidence_loss, int target_k, "
      "int renorm_mode) -> (Tensor, Tensor)");
  m.def(
      "decode(Tensor weights, Tensor ids, Tensor coefficient, Tensor loss, "
      "Tensor norms, int budget) "
      "-> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(dsv4_exfold, CUDA, m) {
  m.impl("prefill", &exfold_prefill_cuda);
  m.impl("decode", &exfold_decode_cuda);
}
