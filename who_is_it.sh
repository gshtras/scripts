#!/bin/bash

# rocm-smi sometimes fail on the bare machine. So, using a container to run the rocm-smi command
container_id=$(docker ps | grep -v CONTAINER | awk '{print $1}' | head -n 1) # using the first container id
pids=$(docker exec -it $container_id rocm-smi --showpids | grep -v "KFD" | grep -v "PID" | grep -v "==========" | awk '{print $1}')

echo $pids

for pid in $pids; do
    if [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]]; then
        echo "Checking PID: $pid"

        container_id=$(for cid in $(docker ps | grep -v CONTAINER | awk '{print $1}') ; do docker top $cid | grep $pid && echo $cid ; done | awk '{print $NF}' | grep -oE '^[a-f0-9]{12}$' )

        if [ -n "$container_id" ]; then
            echo "Container ID for PID $pid: $container_id"
            docker inspect $container_id | grep home
        fi
    fi
done
