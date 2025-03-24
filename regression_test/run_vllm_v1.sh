#!/bin/bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

pip install -r requirements.txt

python v2/run_configs.py --vllm-path ${HOME}/vllm
