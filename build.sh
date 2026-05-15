#!/bin/bash

set -e

if [[ $(pwd) != *"vllm"* ]] ; then
    echo "Must be done in a vllm folder"
    exit 1
fi

if command -v nvidia-smi ; then
    IS_CUDA=1
elif command -v rocm-smi ; then
    IS_ROCM=1
else
    echo "No GPU found"
    exit 1
fi

prefix_arg=
if [[ $(whoami) != "root" ]] ; then
    prefix_arg=" --prefix ~/.local"
fi
if [[ $IS_ROCM == 1 ]] ; then
    if [[ -f requirements-rocm.txt ]] ; then
        pip install -U -r requirements-rocm.txt
    elif [[ -f requirements/rocm.txt ]] ; then
        pip install -U -r requirements/rocm.txt
    else
        echo "No requirements-rocm.txt found"
        exit 1
    fi
    pip install 'ray<2.45'
    pip install 'setuptools<80'
    pip install setuptools_scm
fi
python3 setup.py develop ${prefix_arg}
