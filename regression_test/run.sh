#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

../docker.sh -n regression -c "bash /projects/scripts/regression_test/rebuild_and_run.sh"
