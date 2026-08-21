from __future__ import annotations

from pathlib import Path

from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
BUILD.mkdir(parents=True, exist_ok=True)

load(
    name="libdsv4_exfold_cuda",
    sources=[str(ROOT / "exfold_ops.cpp"), str(ROOT / "exfold_kernels.cu")],
    build_directory=str(BUILD),
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    is_python_module=False,
    verbose=True,
)
