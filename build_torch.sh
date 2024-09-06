#!/bin/bash
set -e
pip uninstall -y torch
git clone https://github.com/pytorch/pytorch
cd pytorch
pip install -r requirements.txt
export CMAKE_PREFIX_PATH=$(python3 -c 'import sys; print(sys.prefix)')
export PYTORCH_ROCM_ARCH=gfx942
python tools/amd_build/build_amd.py
python setup.py develop
