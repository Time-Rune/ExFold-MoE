# ExFold

Official implementation of **ExFold: Unified Expert Folding for Training-Free
MoE Prefill-Decode Acceleration**.

ExFold folds omitted routed experts into retained experts with a calibrated,
directed scalar projection. The repository contains the runtime code, fused
GPU kernels, locked evaluation protocols, and the final calibration artifacts
for Qwen3-30B-A3B and DeepSeek-V4-Flash. Model weights and benchmark data are
downloaded from their original providers.

## Method

For source expert output `u_i`, target output `v_i`, and calibration weight
`w_i`, ExFold fits one scalar for every directed expert pair:

```text
s*(source -> target) = sum_i w_i <u_i, v_i>
                       / (sum_i w_i ||v_i||_2^2 + lambda)
```

At runtime:

- **Prefill:** retain the highest router-score routes for each token. Fold each
  omitted route into its minimum-loss retained target.
- **Decode:** rank experts in the current batch by accumulated
  `router_score * calibrated_output_norm`, retain at most `D`, and remap every
  omitted route to its minimum-loss retained target.
- The routing transformation runs before the standard fused MoE operator, so
  omitted expert outputs are not materialized.

DeepSeek-V4-Flash uses the final unclipped scalar matrix and confidence-aware
prefill transfer. Qwen3 retains the scalar clipping and calibration procedure
used in the paper submission.

## Repository

```text
artifacts/                 final Qwen3 and DeepSeek-V4 calibration matrices
exfold/qwen3/              Qwen3 calibration, Triton kernels, vLLM patch
exfold/deepseek_v4/        DeepSeek-V4 CUDA kernels and vLLM patch
evaluation/                locked OpenCompass protocols and result summaries
scripts/                   serving, quality, and speed reproduction entry points
tests/                     CPU, CUDA, and CUDA Graph correctness checks
```

The two included artifacts are runtime-only tensors. Verify them before use:

```bash
PYTHONPATH=$PWD python scripts/validate_artifacts.py
sha256sum -c artifacts/SHA256SUMS
```

## Requirements

Use Linux, CUDA 12.8 or newer, and Python 3.10+. The reported latency numbers
were measured on NVIDIA H800 GPUs with model weights, code, and benchmark data
on local storage.

Create the matched environments with one command. Qwen3 needs two environments
because the paper prefill and decode operating points use different vLLM
releases:

```bash
bash scripts/install.sh qwen3
bash scripts/install.sh deepseek-v4
# Or install every environment: bash scripts/install.sh all
```

Quality evaluation uses OpenCompass commit
`4df8be67e3b3c67929f7730614c4e5444b62d6a3`. Install it into the active model
environment with:

```bash
PYTHON=.venv-qwen010/bin/python bash scripts/setup_opencompass.sh
```

## Serve

The launchers expose an OpenAI-compatible vLLM endpoint and enable CUDA Graph
for the measured DeepSeek path.

```bash
# Qwen3 paper-quality operating point: P4+D64.
MODEL_PATH=/models/Qwen3-30B-A3B \
PYTHON=.venv-qwen010/bin/python \
bash scripts/serve_qwen3.sh quality exfold

# DeepSeek-V4-Flash report operating point: P3+D64.
MODEL_PATH=/models/DeepSeek-V4-Flash \
PYTHON=.venv-dsv4/bin/python \
bash scripts/serve_deepseek_v4.sh quality exfold
```

Replace `exfold` with `original` for a matched unmodified service. Budgets can
be overridden with `PREFILL_BUDGET` / `DECODE_BUDGET` for Qwen3 and
`EXFOLD_PREFILL_TOPK` / `EXFOLD_DECODE_EXPERT_BUDGET` for DeepSeek-V4.

If Qwen3 TP4 stalls during NCCL initialization on H800, set
`DISABLE_NCCL_NVLS=1`. This was required on the reproduction host and does not
change the model computation.

## Quality Reproduction

One command starts the selected service, waits until it is healthy, evaluates
all paper tasks, and shuts the service down:

```bash
# Qwen3: MATH500, AIME24@32, IFEval, IFBench, GPQA-D,
# LiveCodeBench v5@8, HumanEval+, and MMLU-Pro.
MODEL_PATH=/models/Qwen3-30B-A3B \
PYTHON=.venv-qwen010/bin/python \
bash scripts/reproduce_quality.sh qwen3 exfold

# DeepSeek-V4: IFEval, IFBench, MMLU-Pro, GPQA-D, MATH500 4-shot,
# HumanEval+ avg@8, LiveCodeBench v6, AIME25@8, and AIME26@8.
MODEL_PATH=/models/DeepSeek-V4-Flash \
PYTHON=.venv-dsv4/bin/python \
bash scripts/reproduce_quality.sh deepseek-v4 exfold
```

Run `original` with the same command and environment for the denominator. A
comma-separated third argument evaluates only selected tasks, for example:

```bash
bash scripts/reproduce_quality.sh deepseek-v4 exfold ifeval,ifbench
```

The DeepSeek matrix was calibrated from 64 unlabeled inputs: 56 general
instruction/code/math inputs and 8 benchmark inputs without labels. It is a
transductive, benchmark-aware calibration artifact; it does not modify model
weights. This distinction must be retained when reporting its results.

## Speed Reproduction

The speed launchers always run Original and ExFold with matched requests and
write raw vLLM JSON plus `summary.csv`.

```bash
# Qwen3: 8K prefill QPS 1..8 and 1x256 decode QPS 2..12.
MODEL_PATH=/models/Qwen3-30B-A3B \
QWEN3_PREFILL_PYTHON=$PWD/.venv-qwen010/bin/python \
QWEN3_DECODE_PYTHON=$PWD/.venv-qwen011/bin/python \
DISABLE_NCCL_NVLS=1 \
bash scripts/reproduce_qwen3_speed.sh

# DeepSeek-V4: no-queue 8K TTFT and saturated 1x256 decode.
MODEL_PATH=/models/DeepSeek-V4-Flash \
DEEPSEEK_V4_PYTHON=$PWD/.venv-dsv4/bin/python \
bash scripts/reproduce_deepseek_v4_speed.sh
```

The locked operating points are intentionally phase-specific:

| Model | Metric | Runtime | TP | Workload | ExFold setting |
|---|---|---:|---:|---|---|
| Qwen3 | TTFT | vLLM 0.10.2 V0 | 4 | 8192 in, 1 out, QPS 1-8 | P4 |
| Qwen3 | TPOT | vLLM 0.11.0 V1 | 1 | 1 in, 256 out, QPS 2-12 | calibrated static D32 |
| DeepSeek-V4 | TTFT | vLLM 0.26.0 | 4 | 8192 in, 1 out, concurrency 1 | P3 |
| DeepSeek-V4 | TPOT | vLLM 0.26.0 | 4 | 1 in, 256 out, QPS 64 | D64 |

The Qwen3 quality path uses dynamic batch-level decode selection. Its decode
speed profile freezes the calibrated norm-selected D32 pool so the benchmark
isolates the source-to-target remap and fused-MoE reduction used for the paper
latency result. Do not use that static speed profile to produce quality scores.

## Reproduction Anchors

These are sanity anchors, not hard assertions across drivers or clocks:

| Model and point | Original | ExFold | Speedup |
|---|---:|---:|---:|
| Qwen3, 8K TTFT at QPS 8 (mean) | 472.975 ms | 329.428 ms | 1.436x |
| Qwen3, TPOT at QPS 8 (mean) | 24.228 ms | 10.056 ms | 2.409x |
| DeepSeek-V4, no-queue 8K TTFT (median) | 406.17 ms | 342.58 ms | 1.186x |
| DeepSeek-V4, TPOT at QPS 64 (mean) | 23.509 ms | 18.172 ms | 1.294x |

The locked quality anchors are:

| Qwen3 setting | MATH500 | AIME24 | IFEval | IFBench | GPQA-D | LCB P/A@8 | Eval+ | MMLU-Pro | Avg. | Ret. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original Top8 | 97.40 | 64.48 | 83.73 | 29.31 | 63.13 | 68.26 / 56.89 | 77.44 | 68.57 | 69.04 | 100.00% |
| ExFold P4+D64 | 97.00 | 67.29 | 82.44 | 28.60 | 62.12 | 71.26 / 57.26 | 74.39 | 64.89 | 68.50 | 99.22% |

| DeepSeek-V4 setting | IFEval | IFBench | MMLU-Pro | GPQA-D | MATH500 | Eval+ | LCB v6 | AIME25 | AIME26 | Avg. | Ret. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 82.44 | 36.23 | 81.76 | 73.23 | 93.40 | 89.25 | 54.90 | 70.42 | 72.50 | 72.68 | 100.00% |
| ExFold P3+D64 | 80.96 | 31.56 | 80.21 | 74.75 | 92.60 | 86.97 | 54.38 | 66.67 | 70.83 | 70.99 | 97.68% |

The DeepSeek quality anchor was collected with the same protocol on
`vLLM 0.20.1.dev`; the public launcher targets `vLLM 0.26.0`, which provides
the required DeepSeek-V4 and CUDA Graph paths without local compatibility
backports. Sampling tasks can vary between runs, so compare complete matched
suites rather than individual samples.

## Tests

```bash
PYTHONPATH=$PWD .venv-qwen010/bin/python tests/test_qwen3_calibration.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
  .venv-qwen010/bin/python tests/test_qwen3_kernels.py

PYTHONPATH=$PWD .venv-dsv4/bin/python -m exfold.deepseek_v4.csrc.build
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
  .venv-dsv4/bin/python tests/test_deepseek_v4_cuda.py

PYTHONPATH=$PWD .venv-qwen010/bin/python tests/test_protocols.py
```

## Citation

Please cite the ExFold paper. A final BibTeX entry will be added with the
camera-ready publication metadata.
