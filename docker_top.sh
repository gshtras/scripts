#!/bin/bash
pids=$(/opt/rocm/bin/rocm-smi --showpids | egrep "^[0-9]" | awk '{print $1}')

cids=()

for pid in $pids ; do
    echo "PID: ${pid}"

    new_cids=$(for cid_name in $(docker ps --format "{{.ID}}:{{.Names}}") ; do cid=$(echo $cid_name | cut -d':' -f1) ; name=$(echo $cid_name | cut -d':' -f2) docker top $cid | grep $pid > /dev/null && echo "${cid_name}" ; done)
    for new_cid_name in $new_cids ; do
        new_cid=$(echo $new_cid_name | cut -d':' -f1)
        echo "Container: $new_cid_name"
        cids+=( $new_cid )
    done
done
echo "Result"
sorted_unique_cids=($(echo "${cids[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

for cid in ${sorted_unique_cids[@]} ; do
    echo "Container ID: ${cid}"
    docker inspect $cid | grep home | tr -s ' ' | head -1
done