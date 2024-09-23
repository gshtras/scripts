#!/bin/bash

if [[ $(pwd) != *"vllm"* ]] ; then
    echo "Must be done in a vllm folder"
    exit 1
fi

sudo $(which pip) uninstall -y vllm
pip uninstall -y vllm

rm -rf build
find . -name "*.so" -exec rm -f {} \;
rm -rf .cache
prefix_arg=
if [[ $(whoami) == "gshtrasb" ]] ; then
    prefix_arg=" --prefix ~/.local"
fi
python setup.py develop ${prefix_arg}

