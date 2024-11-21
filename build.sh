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

prefix_arg=
if [[ $(whoami) == "gshtrasb" ]] ; then
    prefix_arg=" --prefix ~/.local"
fi
pip install -r requirements-rocm.txt
python3 setup.py develop ${prefix_arg}
if [[ $gradlib == 1 ]] ; then
    cd gradlib
    python3 setup.py develop ${prefix_arg}
    cd ..
fi

if [[ $cython == 1 ]] ; then
    python3 setup_cython.py build_ext --inplace
fi
