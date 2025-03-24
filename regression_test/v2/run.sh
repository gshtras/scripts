#!/bin/bash

set -e
set -x
echo "Starting $(date)"
export PYTORCH_ROCM_ARCH="gfx942"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../

rm -f $HOME/Projects/docker_kill.log

$SCRIPT_DIR/../docker_top.sh --kill | grep "Killing" | tee $HOME/Projects/docker_kill.log

$SCRIPT_DIR/v2/rebuild_and_run.sh $@ &
pid=$!

function sigint_handler()
{
    echo "Caught SIGINT, killing $pid"
    kill -9 $pid
    ps -alef | grep run_configs | grep -v grep | awk '{print $4}' | xargs kill -9
    docker kill regression
    exit 1
}

trap sigint_handler SIGINT

set +x
while true; do
    sleep 60
    if ! ps -p $pid > /dev/null; then
        break
    fi
    $SCRIPT_DIR/../docker_top.sh -v regression --kill |& grep "Killing" | tee -a $HOME/Projects/docker_kill.log
done

echo "Test finished"
