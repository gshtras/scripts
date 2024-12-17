#!/bin/bash
rocm-smi | grep -v =| grep -v Device |grep % | awk '{printf("%02d:%02d ", $15, $16)}'
