#!/bin/bash

set -e
set -x
echo "Starting $(date)"
export PYTORCH_ROCM_ARCH="gfx942"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

rm -f $HOME/Projects/docker_kill.log

$SCRIPT_DIR/../docker_top.sh --kill | grep "Killing" | tee $HOME/Projects/docker_kill.log

cd $SCRIPT_DIR/../docker
$SCRIPT_DIR/../docker/create.sh
cd -
$SCRIPT_DIR/../docker.sh --noit -n regression -c "bash /projects/scripts/regression_test/rebuild_and_run.sh" &
pid=$!

set +x
while true; do
    sleep 60
    if ! ps -p $pid > /dev/null; then
        break
    fi
    $SCRIPT_DIR/../docker_top.sh -v regression --kill |& grep "Killing" | tee -a $HOME/Projects/docker_kill.log
done

echo "Test finished"
