#!/bin/bash
# Wrapper so vLLM 0.26 finds libcudart.so.13 from the nvidia-cu13 wheel,
# which the CUDA 12.8 host driver does not provide on its own.
set -euo pipefail
ROOT=/workspace/reli81-corpus
export LD_LIBRARY_PATH="${ROOT}/.venv/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "${ROOT}/.venv/bin/reliquary" "$@"
