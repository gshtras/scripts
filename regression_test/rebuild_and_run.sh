#!/bin/bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

cd
git clone https://github.com/rocm/vllm
cd vllm
$SCRIPT_DIR/../rebuild.sh &> /projects/build_regression.log
$SCRIPT_DIR/run_vllm.sh |& tee /projects/result_regression.log
