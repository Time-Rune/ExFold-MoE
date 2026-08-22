"""H20-compatible MHC kernels for DeepSeek-V4 SGLang serving."""

from __future__ import annotations

import os


_INSTALLED = False


def _force_h20_sm90_jit_target() -> None:
    import torch
    from sglang.jit_kernel.utils import arch

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    name = torch.cuda.get_device_name(device)
    if capability != (9, 0) or "H20" not in name:
        return

    # SGLang 0.5.16 marks every Hopper device as architecture-specific 90a.
    # H20 accepts the portable sm_90 image but rejects that sm_90a image.
    arch.get_jit_cuda_arch()
    arch._CUDA_ARCH = arch.ArchInfo(9, 0, "")
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "9.0"
    print("[ExFold SGLang] forced H20 JIT target to sm_90", flush=True)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    _force_h20_sm90_jit_target()
    from exfold_sglang.ops import mhc_split_sinkhorn
    from sglang.kernels.ops.layernorm import mhc

    mhc.hc_split_sinkhorn = mhc_split_sinkhorn
    _INSTALLED = True
    print(
        "[ExFold SGLang] installed CUDA-graph-safe H20 MHC Sinkhorn compatibility",
        flush=True,
    )
