#!/bin/bash
set -e

prefix_arg=
if [[ $(whoami) == "gshtrasb" ]] ; then
    prefix_arg=" --prefix ~/.local"
fi

branch=
if [[ $# -eq 1 ]]; then
    branch=$1
fi

sudo $(which pip) uninstall -y torch
pip uninstall -y torch

git clone https://github.com/pytorch/pytorch
cd pytorch
if [[ -n $branch ]]; then
    git checkout $branch
fi
pip install -r requirements.txt
export CMAKE_PREFIX_PATH=$(python3 -c 'import sys; print(sys.prefix)')
export PYTORCH_ROCM_ARCH=gfx942
git submodule update --init --recursive
python tools/amd_build/build_amd.py
export DEBUG
export CMAKE_BUILD_TYPE=Debug
python setup.py develop ${prefix_arg}
