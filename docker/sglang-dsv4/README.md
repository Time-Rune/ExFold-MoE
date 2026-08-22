# DeepSeek-V4-Flash SGLang image

This image packages the CUDA-graph-safe ExFold adapter and the final unclipped
DeepSeek-V4-Flash calibration artifact. Model weights stay outside the image and
must be mounted at `/workspace/models/DeepSeek-V4-Flash` (or passed with
`--model-path`). HashTopK layers are never folded.

## Launch

Original model:

```bash
exfold-sglang-serve --enable-exfold false
```

ExFold P4+D128:

```bash
exfold-sglang-serve \
  --enable-exfold true \
  --exfold-prefill-k 4 \
  --exfold-decode-k 128
```

The launcher supplies the validated DeepSeek-V4-Flash defaults (`tp=4`, Marlin,
port 8081, tool/reasoning parsers, metrics, cache reporting, and static memory
fraction 0.80). Any ordinary `sglang serve` argument can be appended and
overrides its corresponding default, for example:

```bash
exfold-sglang-serve \
  --enable-exfold true \
  --exfold-prefill-k 4 \
  --exfold-decode-k 128 \
  --port 9000 \
  --mem-fraction-static 0.85
```

Parameter ranges are `prefill K in [1, 6]` and `decode K in [1, 256]`.
