#!/bin/bash

set -e

function run_perf()
{
    set +e
    model=$1
    batch=$2
    in=$3
    out=$4
    tp=$5
    dtype=$6
    shift 6
    eager=
    if [[ $out == "1" ]] ; then
        eager="--enforce-eager"
    fi
    misc=
    if [[ $out != "1" ]] ; then
        misc="--num-scheduler-steps 10"
    fi
    log_name=/projects/tmp/$(echo "${model}" | sed -e 's/\//_/g')_${batch}_${in}_${out}_${tp}.log
    python benchmarks/benchmark_latency.py --enable-chunked-prefill False --load-format dummy --dtype $dtype --num-iters-warmup 2 --num-iters 5 --batch-size $batch $misc --input-len $in --output-len $out --model /models/$model -tp $tp $eager $@ &> $log_name
    echo "${model},${batch},${in},${out},${tp},${dtype},$(cat $log_name | grep "Avg latency:" | awk '{print $3}')"
    set -e
}

function run_corectness()
{
    set +e
    model=$1
    shift
    echo $model
    log_name=/projects/tmp/correctness_$(echo "${model}" | sed -e 's/\//_/g').log
    python /projects/llm_test.py --model /models/$model $@ &> $log_name
    grep "Generated:" $log_name
    set -e
}

function run_vision()
{
    set +e
    model=$1
    shift
    echo $model
    log_name=/projects/tmp/vision_$(echo "${model}" | sed -e 's/\//_/g').log
    VLLM_USE_TRITON_FLASH_ATTN=0 python /projects/llm_test.py --model /models/$model --image-path /projects/image1.jpg --prompt "Describe this image" -tp 4 $@ &> $log_name
    grep "Generated:" $log_name
    set -e
}

function run_p3l()
{
    set +e
    model=$1
    context=$2
    sample=$3
    patch=$4
    shift 4
    echo ${model},${context},${sample},${patch}
    log_name=/projects/tmp/P3L_$(echo "${model}" | sed -e 's/\//_/g')_${batch}_${context}_${sample}_${patch}.log
    python benchmarks/P3L.py --model /models/$model --context-size "$context" --sample-size "$sample" --patch-size $patch $@ &> $log_name
    echo $(cat $log_name |& egrep "Integral Cross|Average Cross|PPL")
    set -e
}

echo $(date +%Y-%m-%d)
pip show vllm |& grep "Version:"

echo "===Correctness==="

run_corectness Meta-Llama-3.1-8B-Instruct
run_corectness Meta-Llama-3.1-405B-Instruct -tp 8
run_corectness mistral-ai-models/Mixtral-8x22B-v0.1/ -tp 4
run_corectness Meta-Llama-3.1-405B-Instruct-FP8-KV -tp 8 --kv-cache-dtype fp8
run_corectness Meta-Llama-3.1-70B-Instruct-FP8-KV -tp 8 --kv-cache-dtype fp8

echo "===Vision==="

run_vision Llama-3.2-90B-Vision-Instruct
run_vision Llama-3.2-90B-Vision-Instruct-FP8-KV

echo "===Performance==="

run_perf llama-2-70b-chat-hf 1 2048 128 8 float16

for batch in 1 16 64 ; do
for in in 128 1024 ; do
for out in 1 128 1024 ; do
for tp in 1 8 ; do
for dtype in float16 bfloat16 ; do
run_perf "Meta-Llama-3.1-8B-Instruct" $batch $in $out $tp $dtype
done
done
done
done
done

for batch in 1 64 ; do
for in in 128 1024 ; do
for out in 1 1024 ; do
for tp in 1 8 ; do
for dtype in float16 bfloat16 ; do
run_perf "Meta-Llama-3.1-70B-Instruct" $batch $in $out $tp $dtype --max-model-len 65536
done
done
done
done
done

run_perf Meta-Llama-3-8B-Instruct 1 2048 2048 8 float16
run_perf Meta-Llama-3-8B-Instruct 64 256 256 8 float16

run_perf Meta-Llama-3.1-405B-Instruct 1 128 1024 8 float16
run_perf Meta-Llama-3.1-405B-Instruct 16 1024 1024 8 float16

run_perf mistral-ai-models/Mixtral-8x22B-v0.1/ 16 1024 1024 8 float16

run_perf Meta-Llama-3.1-405B-Instruct-FP8-KV 16 1024 1024 8 float16
run_perf Meta-Llama-3.1-70B-Instruct-FP8-KV 16 1024 1024 8 float16

echo "===P3L==="

run_p3l Meta-Llama-3.1-8B-Instruct 1024 512 10 --dtype float16
run_p3l Meta-Llama-3.1-70B-Instruct-FP8-KV 1024 512 10 --kv-cache-dtype fp8 --dtype float16
