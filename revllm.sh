#!/bin/bash

repo=https://github.com/vllm-project/vllm.git
branch=main
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
  --repo)
    repo="$2"
    shift
    ;;
  --branch)
    branch="$2"
    shift
    ;;
  --rocm)
    repo=https://github.com/ROCm/vllm.git
    ;;
  *)
    echo "Unknown option: $key"
    exit 1
    ;;
  esac
  shift
done

set -e
set -o pipefail

if [[ -z "$repo" ]]; then
  echo "Error: --repo is required"
  exit 1
fi

if [[ -z "$branch" ]]; then
  echo "Error: --branch is required"
  exit 1
fi

cd

rm -rf vllm
git clone "$repo"
cd vllm
git checkout "$branch"
${SCRIPT_DIR}/rebuild.sh
