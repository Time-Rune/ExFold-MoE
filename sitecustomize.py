"""Install the selected ExFold runtime patch before model modules are imported."""

import os


if os.environ.get("EXFOLD_SGLANG_H20_COMPAT", "0") == "1":
    from exfold_sglang.mhc_compat import install as install_mhc_compat

    install_mhc_compat()

if os.environ.get("EXFOLD_SGLANG_ENABLE", "0") == "1":
    from exfold_sglang.runtime import install as install_sglang

    install_sglang()


model = os.environ.get("EXFOLD_MODEL", "").strip().lower()
enabled = os.environ.get("EXFOLD_ENABLE", "0") == "1"

if enabled and model == "qwen3":
    from exfold.qwen3.vllm_patch import patch_vllm_qwen3_moe

    patch_vllm_qwen3_moe()
elif enabled and model in {"deepseek-v4", "deepseek_v4"}:
    from exfold.deepseek_v4 import vllm_patch  # noqa: F401
elif enabled:
    raise RuntimeError(f"Unsupported EXFOLD_MODEL={model!r}")
