#!/bin/bash
set -e
pip uninstall -y torch
git clone https://github.com/pytorch/pytorch
cd pytorch
git checkout 8bc4033
pip install -r requirements.txt
export CMAKE_PREFIX_PATH=$(python3 -c 'import sys; print(sys.prefix)')
export PYTORCH_ROCM_ARCH=gfx942
git submodule update --init --recursive
python tools/amd_build/build_amd.py
export DEBUG
export CMAKE_BUILD_TYPE=Debug
python setup.py develop
