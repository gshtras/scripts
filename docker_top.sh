#!/bin/bash

pids=$(/opt/rocm/bin/rocm-smi --showpids | egrep "^[0-9]" | awk '{print $1}')

cids=()

for pid in $pids ; do
    echo "PID: ${pid}"

    new_cids=$(for cid in $(docker ps | grep -v CONTAINER | awk '{print $1}') ; do docker top $cid | grep $pid > /dev/null && echo $cid ; done)
    for new_cid in $new_cids ; do
        echo "Container: $new_cid"
        cids+=( $new_cid )
    done
done
echo "Result"

sorted_unique_cids=($(echo "${cids[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

for cid in ${sorted_unique_cids[@]} ; do
    echo "Container ID: ${cid}"
    docker inspect $cid | grep home | head -1
done