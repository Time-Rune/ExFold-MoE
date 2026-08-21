"""Install the selected ExFold vLLM patch before model modules are imported."""

import os


model = os.environ.get("EXFOLD_MODEL", "").strip().lower()
enabled = os.environ.get("EXFOLD_ENABLE", "0") == "1"

if enabled and model == "qwen3":
    from exfold.qwen3.vllm_patch import patch_vllm_qwen3_moe

    patch_vllm_qwen3_moe()
elif enabled and model in {"deepseek-v4", "deepseek_v4"}:
    from exfold.deepseek_v4 import vllm_patch  # noqa: F401
elif enabled:
    raise RuntimeError(f"Unsupported EXFOLD_MODEL={model!r}")
