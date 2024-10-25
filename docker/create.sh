#!/bin/bash
set -e
set -x

wget 'https://raw.githubusercontent.com/ROCm/vllm/refs/heads/main/requirements-lint.txt'

docker build -f Dockerfile.vllm \
    --build-arg "UID=$(id -u)" \
    --build-arg "GID=$(id -g)" \
    --build-arg "RENDER_GID=$(cat /etc/group | grep render | cut -d: -f3)" \
    --build-arg "PYTORCH_ROCM_ARCH=$(/opt/rocm/bin/offload-arch)" \
    -t greg_vllm .

rm -f requirements-lint.txt
