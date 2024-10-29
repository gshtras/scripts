#!/bin/bash

if [[ $(pwd) != *"vllm"* ]] ; then
    echo "Must be done in a vllm folder"
    exit 1
fi

prefix_arg=
if [[ $(whoami) == "gshtrasb" ]] ; then
    prefix_arg=" --prefix ~/.local"
fi
python setup.py develop ${prefix_arg}

