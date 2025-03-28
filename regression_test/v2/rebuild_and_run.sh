#!/bin/bash

set -e
set -x

branch=main
args=
base=rocm/vllm-dev:base

while [[ $# -gt 0 ]] ; do
  i=$1
  case $i in
  --base)
    base="$2"
    shift
  ;;
  --branch)
    branch="$2"
    shift
  ;;
  -s|--short)
    args="$args --short-run"
  ;;
  -v|--verbose)
    args="$args --verbose"
  ;;
  --no-dashboard)
    args="$args --no-dashboard"
  ;;
  *)
    echo "Unrecognized argument: $1"
    exit 1
  ;;
  esac
  shift
done

export PYTORCH_ROCM_ARCH="gfx942"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../

cd $HOME/Projects/tmp
rm -rf vllm
git clone https://github.com/ROCm/vllm
cd vllm
git checkout $branch
docker build --no-cache -f Dockerfile.rocm --build-arg BASE_IMAGE=$base --build-arg ARG_PYTORCH_ROCM_ARCH=gfx942 --build-arg USE_CYTHON=1 -t vllm_regression_raw .
 
cd $SCRIPT_DIR/../docker
$SCRIPT_DIR/../docker/create.sh -b vllm_regression_raw -n vllm_regression
cd -

python3 $SCRIPT_DIR/v2/run_configs.py $args
