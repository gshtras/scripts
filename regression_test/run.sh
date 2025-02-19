#!/bin/bash

set -e
set -x
echo "Starting $(date)"
export PYTORCH_ROCM_ARCH="gfx942"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

$SCRIPT_DIR/../docker_top.sh --kill

cd $SCRIPT_DIR/../docker
$SCRIPT_DIR/../docker/create.sh
cd -
$SCRIPT_DIR/../docker.sh --noit -n regression -c "bash /projects/scripts/regression_test/rebuild_and_run.sh" &
pid=$!

while true; do
    sleep 60
    if ! ps -p $pid > /dev/null; then
        break
    fi
    $SCRIPT_DIR/../docker_top.sh -v regression --kill
done
