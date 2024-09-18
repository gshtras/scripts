#!/bin/bash
set -e
git clone https://github.com/ROCm/hipBLAS-common.git
cd hipBLAS-common
mkdir build
cd build
cmake ..
make package
dpkg -i ./*.deb
cd ../..

git clone https://github.com/ROCm/hipBLASLt
cd hipBLASLt
SCCACHE_IDLE_TIMEOUT=1800 ./install.sh -idc --architecture gfx942
