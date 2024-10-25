#!/bin/bash
set -e
set -x

rm -f requirements-lint.txt* bundle.tgz

wget 'https://raw.githubusercontent.com/ROCm/vllm/refs/heads/main/requirements-lint.txt'

if [[ -e ~/Projects/docker_bundle.tgz ]] ; then
    cp ~/Projects/docker_bundle.tgz bundle.tgz
else
    tar czvf bundle.tgz --files-from=/dev/null
fi

docker build -f Dockerfile.vllm \
    --build-arg "UID=$(id -u)" \
    --build-arg "GID=$(id -g)" \
    --build-arg "USERNAME=$(whoami)" \
    --build-arg "RENDER_GID=$(cat /etc/group | grep render | cut -d: -f3)" \
    --build-arg "PYTORCH_ROCM_ARCH=$(/opt/rocm/bin/offload-arch)" \
    -t greg_vllm .

rm -f requirements-lint.txt bundle.tgz
