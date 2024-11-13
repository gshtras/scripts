#!/bin/bash

if [[ $(pwd) != *"vllm"* ]] ; then
    echo "Must be done in a vllm folder"
    exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

sudo $(which pip) uninstall -y vllm
pip uninstall -y vllm

rm -rf build
find . -name "*.so" -exec rm -f {} \;
rm -rf .cache

${SCRIPT_DIR}/build.sh $@
