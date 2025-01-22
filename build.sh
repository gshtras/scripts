#!/bin/bash

set -e

if [[ $(pwd) != *"vllm"* ]] ; then
    echo "Must be done in a vllm folder"
    exit 1
fi
gradlib=
cython=

while [[ $# -gt 0 ]] ; do
  i=$1
  case $i in
  --gradlib)
    gradlib=1
  ;;
  --cython)
    cython=1
  ;;
  esac
  shift
done

if command -v rocm-smi ; then
    IS_ROCM=1
elif command -v nvidia-smi ; then
    IS_CUDA=1
else
    echo "No GPU found"
    exit 1
fi

prefix_arg=
if [[ $(whoami) == "gshtrasb" ]] ; then
    prefix_arg=" --prefix ~/.local"
fi
if [[ $IS_ROCM == 1 ]] ; then
    pip install -r requirements-rocm.txt
fi
python3 setup.py develop ${prefix_arg}
if [[ $gradlib == 1 ]] ; then
    cd gradlib
    python3 setup.py develop ${prefix_arg}
    cd ..
fi

if [[ $cython == 1 ]] ; then
    python3 setup_cython.py build_ext --inplace
fi
