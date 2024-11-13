#!/bin/bash

set -e

if [[ $(pwd) != *"vllm"* ]] ; then
    echo "Must be done in a vllm folder"
    exit 1
fi

prefix_arg=
if [[ $(whoami) == "gshtrasb" ]] ; then
    prefix_arg=" --prefix ~/.local"
fi
pip install -r requirements-rocm.txt
python setup.py develop ${prefix_arg}
if [[ "$1" == "--gradlib" ]] ; then
    cd gradlib
    python setup.py develop ${prefix_arg}
fi
