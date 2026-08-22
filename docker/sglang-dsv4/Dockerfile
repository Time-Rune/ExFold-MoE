ARG BASE_IMAGE=artifactory.devops.xiaohongshu.com/media/sglang:v0.5.16-cu129-dspark-grammarfix-20260819
FROM ${BASE_IMAGE}

WORKDIR /opt/exfold
COPY exfold_sglang /opt/exfold/exfold_sglang
COPY sitecustomize.py /opt/exfold/sitecustomize.py
COPY artifacts/deepseek-v4-flash.pt /opt/exfold/artifacts/deepseek-v4-flash.pt
COPY docker/sglang-dsv4/exfold-sglang-serve /usr/local/bin/exfold-sglang-serve

ENV EXFOLD_HOME=/opt/exfold \
    PYTHONPATH=/opt/exfold \
    PATH=/usr/local/cuda/bin:${PATH}

RUN echo "8cb5fb81f4439da72d2cfb72d5cbf180107a8c0bd9895e18bfc46a1063108d7f  artifacts/deepseek-v4-flash.pt" | sha256sum -c - && \
    TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=16 \
      python3 -m exfold_sglang.cuda_ops.build && \
    chmod 0755 /usr/local/bin/exfold-sglang-serve && \
    python3 -m exfold_sglang.entrypoint \
      --enable-exfold false --exfold-print-command >/tmp/original-command.txt && \
    python3 -m exfold_sglang.entrypoint \
      --enable-exfold true --exfold-prefill-k 4 --exfold-decode-k 128 \
      --exfold-print-command >/tmp/exfold-command.txt

ENTRYPOINT ["exfold-sglang-serve"]
