#!/bin/bash
set -e
set -x

name=${USER}_vllm
base=rocm/vllm-dev:base

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

wget 'https://raw.githubusercontent.com/ROCm/vllm/refs/heads/main/requirements-lint.txt'

if [[ -e ~/Projects/docker_bundle.tgz ]] ; then
    cp ~/Projects/docker_bundle.tgz bundle.tgz
else
    tar czvf bundle.tgz --files-from=/dev/null
fi

USER=${USER:-$(whoami)}
PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-$(/opt/rocm/bin/offload-arch)}

docker build -f Dockerfile.vllm \
    --build-arg "BASE_IMAGE=${base}" \
    --build-arg "UID=$(id -u)" \
    --build-arg "GID=$(id -g)" \
    --build-arg "USERNAME=$(whoami)" \
    --build-arg "RENDER_GID=$(cat /etc/group | grep render | cut -d: -f3)" \
    --build-arg "PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}" \
    -t ${name} .

rm -f requirements-lint.txt bundle.tgz
