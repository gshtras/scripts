#!/bin/bash
set -e
set -x

branch=
if [[ $# -eq 1 ]]; then
    branch=$1
fi

prefix_arg=
if [[ $(whoami) == "gshtrasb" ]] ; then
    prefix_arg=" --prefix ~/.local"
fi

sudo $(which pip) uninstall -y triton
pip uninstall -y triton

git clone https://github.com/triton-lang/triton.git
cd triton
if [[ -n $branch ]]; then
    git checkout $branch
fi
cd python
python setup.py develop ${prefix_arg}
