#!/bin/bash
docker run -it --rm --device=/dev/kfd --device=/dev/dri --mount type=bind,source=/home/gshtrasb/Projects,target=/projects --mount type=bind,source=/data/models,target=/models --group-add video --cap-add=SYS_PTRACE $@
