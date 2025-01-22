#!/bin/bash
set -e
set -x

USER=${USER:-$(whoami)}
name=${USER}_vllm

if command -v rocm-smi ; then
    IS_ROCM=1
    vllm_repo=ROCm
    base=rocm/vllm-dev:base
elif command -v nvidia-smi ; then
    IS_CUDA=1
    vllm_repo=vllm-project
    # From vllm-project/vllm.git
    # docker build --target dev -t vllm_dev .
    base=vllm_dev
else
    echo "No GPU found"
    exit 1
fi


while [[ $# -gt 0 ]] ; do
  i=$1
  case $i in
  -n|--name)
    name="$2"
    shift
  ;;
  -b|--base)
    base="$2"
    shift
  ;;
  *)
    echo "Unrecognized argument: $1"
    exit 1
  ;;
  esac
  shift
done

rm -f requirements-lint.txt* bundle.tgz

wget "https://raw.githubusercontent.com/${vllm_repo}/vllm/refs/heads/main/requirements-lint.txt"

if [[ -e ~/Projects/docker_bundle.tgz ]] ; then
    cp ~/Projects/docker_bundle.tgz bundle.tgz
else
    tar czvf bundle.tgz --files-from=/dev/null
fi

if [[ $IS_ROCM == 1 ]] ; then
    PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-$(/opt/rocm/bin/offload-arch)}
elif [[ $IS_CUDA == 1 ]] ; then
    TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv | tail -1)
fi

docker build -f Dockerfile.vllm \
    --build-arg "BASE_IMAGE=${base}" \
    --build-arg "UID=$(id -u)" \
    --build-arg "GID=$(id -g)" \
    --build-arg "USERNAME=$(whoami)" \
    --build-arg "RENDER_GID=$(cat /etc/group | grep render | cut -d: -f3)" \
    --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" \
    --build-arg "PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}" \
    --build-arg "IS_ROCM=${IS_ROCM}" \
    -t ${name} .

rm -f requirements-lint.txt bundle.tgz
