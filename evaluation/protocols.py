"""Locked OpenCompass protocols used for the ExFold paper and report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


OPENCOMPASS_COMMIT = "4df8be67e3b3c67929f7730614c4e5444b62d6a3"


@dataclass(frozen=True)
class Task:
    module: str
    symbol: str
    abbr: str
    max_seq_len: int
    max_out_len: int
    thinking: bool
    temperature: float
    top_p: float
    transform: str = "plain"
    repeats: Optional[str] = None
    single_dataset: bool = False


QWEN3_TASKS = {
    "math500": Task(
        "math.math_500_gen", "math_datasets", "math-500", 40960, 38912,
        True, 0.6, 0.95,
    ),
    "aime24": Task(
        "aime2024.aime2024_0shot_nocot_gen_2b9dc2",
        "aime2024_datasets", "aime2024", 40960, 38912, True, 0.6, 0.95,
        transform="math_repeat", repeats="aime",
    ),
    "ifeval": Task(
        "IFEval.IFEval_gen", "ifeval_datasets", "IFEval", 8192, 1280,
        False, 0.0, 1.0, transform="strip_reasoning",
    ),
    "ifbench": Task(
        "IFBench.IFBench_gen", "ifbench_datasets", "IFBench", 8192, 4096,
        False, 0.7, 0.8, transform="strip_reasoning",
    ),
    "gpqa": Task(
        "gpqa.gpqa_openai_simple_evals_gen_5aeece", "gpqa_datasets",
        "GPQA_diamond", 40960, 32768, True, 0.6, 0.95,
    ),
    "mmlu_pro": Task(
        "mmlu_pro.mmlu_pro_gen", "mmlu_pro_datasets", "mmlu_pro", 8192,
        4096, True, 0.6, 0.95, transform="mmlu_pro",
    ),
    "humaneval_plus": Task(
        "humaneval_plus.humaneval_plus_gen_8e312c",
        "humaneval_plus_datasets", "humaneval_plus", 8192, 4096, True,
        0.6, 0.95,
    ),
    "lcb_v5": Task(
        "livecodebench.livecodebench_time_split_gen_a4f90b",
        "LCBCodeGeneration_dataset", "lcb_code_generation_v5", 40960,
        20000, True, 0.6, 0.95, transform="code_repeat", repeats="code",
        single_dataset=True,
    ),
}


DEEPSEEK_V4_TASKS = {
    "ifeval": Task(
        "IFEval.IFEval_gen", "ifeval_datasets", "IFEval", 8192, 1280,
        False, 0.0, 1.0, transform="strip_reasoning",
    ),
    "ifbench": Task(
        "IFBench.IFBench_gen", "ifbench_datasets", "IFBench", 8192, 4096,
        False, 1.0, 1.0, transform="strip_reasoning",
    ),
    "mmlu_pro": Task(
        "mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de", "mmlu_pro_datasets",
        "mmlu_pro", 8192, 8192, False, 1.0, 1.0,
        transform="mmlu_pro",
    ),
    "gpqa": Task(
        "gpqa.gpqa_0shot_nocot_gen_772ea0", "gpqa_datasets",
        "GPQA_diamond", 8192, 8192, False, 1.0, 1.0,
    ),
    "math500": Task(
        "math.math_500_gen", "math_datasets", "math-500-minerva-4shot",
        40000, 32768, True, 1.0, 1.0, transform="math500_4shot",
    ),
    "humaneval_plus": Task(
        "humaneval_plus.humaneval_plus_gen_8e312c",
        "humaneval_plus_datasets", "humaneval_plus", 40000, 16384, True,
        1.0, 1.0, transform="code_repeat", repeats="code",
    ),
    "lcb_v6": Task(
        "livecodebench.livecodebench_v6_academic",
        "LCBCodeGeneration_dataset", "lcb_code_generation_v6", 32768, 8192,
        False, 1.0, 1.0, transform="code_repeat", repeats="one",
        single_dataset=True,
    ),
    "aime25": Task(
        "aime2025.aime2025_cascade_eval_gen_5e9f4f",
        "aime2025_datasets", "aime2025", 40000, 32768, True, 1.0, 1.0,
        transform="math_repeat", repeats="aime",
    ),
    "aime26": Task(
        "aime2026.aime2026_cascade_eval_gen_6ff468",
        "aime2026_datasets", "aime2026", 40000, 32768, True, 1.0, 1.0,
        transform="math_repeat", repeats="aime",
    ),
}


PROTOCOLS = {
    "qwen3": QWEN3_TASKS,
    "deepseek-v4": DEEPSEEK_V4_TASKS,
}

PAPER_TASKS = {
    "qwen3": tuple(QWEN3_TASKS),
    "deepseek-v4": tuple(DEEPSEEK_V4_TASKS),
}

DEFAULTS = {
    "qwen3": dict(batch_size=16, max_workers=16, infer_workers=1, qps=32.0),
    "deepseek-v4": dict(
        batch_size=4096, max_workers=96, infer_workers=1, qps=256.0
    ),
}

MMLU_PRO_SUBSETS = tuple(
    "mmlu_pro_" + category
    for category in (
        "math", "physics", "chemistry", "law", "engineering", "other",
        "economics", "health", "psychology", "business", "biology",
        "philosophy", "computer_science", "history",
    )
)
