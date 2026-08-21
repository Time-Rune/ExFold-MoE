#!/usr/bin/env python3
"""Run the locked Qwen3 or DeepSeek-V4-Flash quality protocol."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.protocols import (
    DEFAULTS,
    MMLU_PRO_SUBSETS,
    OPENCOMPASS_COMMIT,
    PAPER_TASKS,
    PROTOCOLS,
    Task,
)


def parse_tasks(protocol: str, value: str) -> list[str]:
    names: list[str] = []
    for item in value.split(","):
        item = item.strip()
        names.extend(PAPER_TASKS[protocol] if item == "paper" else (item,))
    unknown = sorted(set(names) - set(PROTOCOLS[protocol]))
    if unknown:
        raise ValueError(f"Unsupported {protocol} tasks: {', '.join(unknown)}")
    return list(dict.fromkeys(names))


def ensure_opencompass_checkout(path: Path) -> None:
    run_py = path / "run.py"
    if not run_py.is_file():
        raise FileNotFoundError(f"OpenCompass entry point not found: {run_py}")
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Cannot inspect OpenCompass checkout: {path}") from error
    if revision != OPENCOMPASS_COMMIT:
        raise RuntimeError(
            "OpenCompass revision mismatch: "
            f"expected {OPENCOMPASS_COMMIT}, found {revision}"
        )


def ensure_humaneval_plus_compat(opencompass_dir: Path) -> None:
    """Backport EvalPlus 0.3 result parsing to the pinned OpenCompass commit."""
    path = opencompass_dir / "opencompass/datasets/humaneval.py"
    source = path.read_text(encoding="utf-8")
    interface_marker = "from evalplus.evaluate import PASS, estimate_pass_at_k"
    subset_marker = "selected_references = set(references)"
    if interface_marker in source and subset_marker in source:
        return
    old = """            score = evaluate(flags)
            results_path = osp.join(tmp_dir, 'human_eval_eval_results.json')
            with open(results_path, 'r') as f:
                results = json.load(f)
            details = {}
            for index in range(len(predictions)):
                r = results['eval'][references[index]]

                details[str(index)] = {
                    'prompt': prompts[index],
                    'prediction': predictions[index],
                    'reference': references[index],
                    'base_result': r['base'][0][0],
                    'plus_result': r['plus'][0][0],
                    'is_correct': r['base'][0][0] == 'success' and r['plus'][0][0] == 'success',
                }
                if r['nfiles'] > 1:
                    details[str(index)]['warning'] = 'Multiple files in the solution. Details may be wrong.'
        results = {f'humaneval_plus_{k}': score[k] * 100 for k in score}
"""
    new = """            try:
                score = evaluate(**flags)
            except TypeError:
                score = evaluate(flags)
            results_path = osp.join(tmp_dir, 'human_eval_eval_results.json')
            with open(results_path, 'r') as f:
                results = json.load(f)
            if score is None:
                from evalplus.evaluate import PASS, estimate_pass_at_k
                import numpy as np
                total = np.array([len(r) for r in results['eval'].values()])
                correct = np.array([
                    sum(x.get('base_status') == PASS and
                        x.get('plus_status') == PASS for x in rows)
                    for rows in results['eval'].values()
                ])
                score = {
                    f'pass@{k}': estimate_pass_at_k(total, correct, k).mean()
                    for k in self.k if (total >= k).all()
                }
            details = {}
            for index in range(len(predictions)):
                r = results['eval'][references[index]]
                if isinstance(r, list) and r and 'base_status' in r[0]:
                    base_result = r[0]['base_status']
                    plus_result = r[0].get('plus_status')
                    nfiles = len(r)
                else:
                    base_result = r['base'][0][0]
                    plus_result = r['plus'][0][0]
                    nfiles = r['nfiles']
                details[str(index)] = {
                    'prompt': prompts[index],
                    'prediction': predictions[index],
                    'reference': references[index],
                    'base_result': base_result,
                    'plus_result': plus_result,
                    'is_correct': (
                        base_result in ['success', 'pass']
                        and plus_result in ['success', 'pass']),
                }
                if nfiles > 1:
                    details[str(index)]['warning'] = 'Multiple files in the solution. Details may be wrong.'
        results = {
            f"humaneval_plus_{str(k).split('@')[-1]}": v * 100
            for k, v in score.items()
        }
"""
    if interface_marker not in source:
        if old not in source:
            raise RuntimeError("Unsupported OpenCompass HumanEval+ evaluator")
        source = source.replace(old, new)
    subset_old = """            try:
                score = evaluate(**flags)
            except TypeError:
                score = evaluate(flags)
"""
    subset_new = """            import importlib
            evalplus_module = importlib.import_module('evalplus.evaluate')
            load_all_problems = evalplus_module.get_human_eval_plus
            selected_references = set(references)
            evalplus_module.get_human_eval_plus = lambda **kwargs: {
                task_id: problem
                for task_id, problem in load_all_problems(**kwargs).items()
                if task_id in selected_references
            }
            try:
                try:
                    score = evaluate(**flags)
                except TypeError:
                    score = evaluate(flags)
            finally:
                evalplus_module.get_human_eval_plus = load_all_problems
"""
    if subset_marker not in source:
        if subset_old not in source:
            raise RuntimeError("Cannot add subset support to HumanEval+")
        source = source.replace(subset_old, subset_new)
    path.write_text(source, encoding="utf-8")


def ensure_nltk() -> None:
    try:
        import nltk
    except ImportError as error:
        raise RuntimeError("IFEval and IFBench require nltk") from error
    try:
        nltk.data.find("tokenizers/punkt_tab/english")
    except LookupError:
        if not nltk.download("punkt_tab", quiet=True):
            raise RuntimeError("Cannot download NLTK punkt_tab")


def task_repeats(args: argparse.Namespace, task: Task) -> int:
    if task.repeats == "aime":
        return args.aime_repeats
    if task.repeats == "code":
        return args.code_repeats
    if task.repeats == "one":
        return 1
    return 1


def transform_block(task: Task, repeats: int) -> str:
    lines: list[str] = []
    if task.transform == "math500_4shot":
        lines.extend(
            [
                "    dataset['abbr'] = 'math-500-minerva-4shot'",
                "    dataset['infer_cfg'] = dict(",
                "        prompt_template=dict(type=PromptTemplate, template=dict(",
                "            round=[dict(role='HUMAN', prompt=math_4shot_prompt + "
                "'\\n\\nProblem:\\n{problem}\\n\\nSolution:')])),",
                "        retriever=dict(type=ZeroRetriever),",
                "        inferencer=dict(type=GenInferencer),",
                "    )",
                "    dataset['eval_cfg'] = dict(",
                "        evaluator=dict(type=MATHVerifyEvaluator))",
            ]
        )
    elif task.transform == "math_repeat":
        lines.extend(
            [
                f"    dataset['n'] = {repeats}",
                f"    dataset['abbr'] = {f'{task.abbr}_repeat_{repeats}'!r}",
                "    dataset['eval_cfg'] = dict(",
                "        evaluator=dict(type=MATHVerifyEvaluator))",
            ]
        )
    elif task.transform == "code_repeat":
        lines.extend(
            [
                f"    dataset['n'] = {repeats}",
                f"    dataset['abbr'] = {f'{task.abbr}_repeat_{repeats}'!r}",
                "    dataset.setdefault('eval_cfg', {})['k'] = [1]",
            ]
        )
    elif task.transform == "strip_reasoning":
        lines.append(
            "    dataset.setdefault('eval_cfg', {})['pred_postprocessor'] = "
            "dict(type=extract_non_reasoning_content)"
        )
    return "\n".join(lines)


def summary(task: Task, repeats: int) -> tuple[str, str]:
    abbr = task.abbr
    if task.repeats:
        abbr = f"{abbr}_repeat_{repeats}"
    groups = "[]"
    if task.transform == "mmlu_pro":
        entry = "['mmlu_pro', 'naive_average']"
        groups = repr([dict(name="mmlu_pro", subsets=list(MMLU_PRO_SUBSETS))])
    else:
        entry = repr(abbr)
    return entry, groups


def write_config(args: argparse.Namespace, name: str, output: Path) -> None:
    task = PROTOCOLS[args.protocol][name]
    repeats = task_repeats(args, task)
    dataset_expression = f"[{task.symbol}]" if task.single_dataset else task.symbol
    extra_imports = ""
    if task.transform == "math500_4shot":
        extra_imports = """
from opencompass.evaluator import MATHVerifyEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
"""
    elif task.transform == "math_repeat":
        extra_imports = "from opencompass.evaluator import MATHVerifyEvaluator\n"
    summary_entry, summary_groups = summary(task, repeats)
    test_range = args.test_range or (f"[:{args.limit}]" if args.limit else None)
    test_range_line = (
        f"    dataset.setdefault('reader_cfg', {{}})['test_range'] = {test_range!r}"
        if test_range
        else ""
    )
    seed_line = f"        seed={args.seed},\n" if args.seed is not None else ""
    thinking_mode = (
        f"            thinking_mode={'thinking' if task.thinking else 'chat'!r},\n"
        if args.protocol == "deepseek-v4"
        else ""
    )
    model_postprocessor = (
        "    pred_postprocessor=dict(type=extract_non_reasoning_content),\n"
        if task.thinking
        else ""
    )
    math_prompt_import = (
        "    from opencompass.configs.datasets.math."
        "math_4shot_example_from_google_research import prompt as math_4shot_prompt"
        if task.transform == "math500_4shot"
        else ""
    )
    output.write_text(
        f"""from mmengine.config import read_base
from opencompass.models import OpenAISDK
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask
from opencompass.utils.text_postprocessors import extract_non_reasoning_content
{extra_imports}
with read_base():
    from opencompass.configs.datasets.{task.module} import {task.symbol}
{math_prompt_import}

datasets = {dataset_expression}
for dataset in datasets:
    dataset.setdefault('infer_cfg', {{}}).setdefault('inferencer', {{}})[
        'max_out_len'] = {task.max_out_len}
{transform_block(task, repeats)}
{test_range_line}

models = [dict(
    type=OpenAISDK,
    abbr={args.model_name!r},
    path={args.model_name!r},
    key='EMPTY',
    openai_api_base={args.api_base!r},
    tokenizer_path={str(args.model_path)!r},
    max_seq_len={task.max_seq_len},
    max_out_len={task.max_out_len},
    batch_size={args.batch_size},
    max_workers={args.max_workers},
    query_per_second={args.qps},
    retry={args.retry},
    timeout={args.timeout},
    temperature={task.temperature},
    extra_body=dict(
        top_p={task.top_p},
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
{seed_line}        chat_template_kwargs=dict(
            enable_thinking={task.thinking},
{thinking_mode}        ),
    ),
{model_postprocessor}    run_cfg=dict(num_gpus=0),
)]

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker={args.infer_workers},
        force_rebuild=True,
        dataset_size_path={str(output.parent / 'dataset_size.json')!r},
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers={args.infer_workers},
        task=dict(type=OpenICLInferTask),
    ),
)
eval = dict(
    partitioner=dict(type=NaivePartitioner, n={args.eval_workers}),
    runner=dict(
        type=LocalRunner,
        max_num_workers={args.eval_workers},
        task=dict(type=OpenICLEvalTask),
    ),
)
summarizer = dict(
    dataset_abbrs=[{summary_entry}],
    summary_groups={summary_groups},
)
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), required=True)
    parser.add_argument("--opencompass-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--tasks", default="paper")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--infer-workers", type=int)
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--qps", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument("--aime-repeats", type=int)
    parser.add_argument("--code-repeats", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--test-range")
    parser.add_argument(
        "--mode", choices=("all", "infer", "eval", "viz"), default="all"
    )
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def apply_defaults(args: argparse.Namespace) -> None:
    defaults = DEFAULTS[args.protocol]
    args.model_name = args.model_name or (
        "qwen3-30b-a3b" if args.protocol == "qwen3" else "deepseek-v4-flash"
    )
    args.work_dir = args.work_dir or Path(f"results/{args.protocol}-quality")
    args.batch_size = args.batch_size or defaults["batch_size"]
    args.max_workers = args.max_workers or defaults["max_workers"]
    args.infer_workers = args.infer_workers or defaults["infer_workers"]
    args.qps = args.qps or defaults["qps"]
    args.aime_repeats = args.aime_repeats or (
        32 if args.protocol == "qwen3" else 8
    )
    args.code_repeats = args.code_repeats or 8


def latest_run(task_dir: Path) -> str:
    candidates = sorted(path.name for path in task_dir.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No reusable result under {task_dir}")
    return candidates[-1]


def main() -> None:
    args = parse_args()
    apply_defaults(args)
    ensure_opencompass_checkout(args.opencompass_dir)
    if args.limit is not None and args.test_range is not None:
        raise ValueError("--limit and --test-range are mutually exclusive")
    if args.test_range and not re.fullmatch(r"\[\d*:\d*\]", args.test_range):
        raise ValueError("--test-range must have the form [start:end]")
    if args.aime_repeats <= 0 or args.code_repeats <= 0:
        raise ValueError("repeat counts must be positive")
    names = parse_tasks(args.protocol, args.tasks)
    if "humaneval_plus" in names:
        ensure_humaneval_plus_compat(args.opencompass_dir)
    if {"ifeval", "ifbench"}.intersection(names) and not args.dry_run:
        ensure_nltk()

    for name in names:
        task_dir = args.work_dir / name
        task_dir.mkdir(parents=True, exist_ok=True)
        config = task_dir / "config.py"
        write_config(args, name, config)
        print(config, flush=True)
        if args.dry_run:
            continue
        command = [
            sys.executable,
            str(args.opencompass_dir / "run.py"),
            str(config),
            "-w",
            str(task_dir),
            "--mode",
            args.mode,
        ]
        if args.reuse:
            command.extend(["--reuse", latest_run(task_dir)])
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            [
                str(PROJECT_ROOT),
                str(args.opencompass_dir),
                environment.get("PYTHONPATH", ""),
            ]
        ).rstrip(os.pathsep)
        subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
