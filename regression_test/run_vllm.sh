#!/bin/bash

set -e

function run_perf()
{
    model=$1
    batch=$2
    in=$3
    out=$4
    tp=$5
    eager=
    if [[ $out == "1" ]] ; then
        eager="--enforce-eager"
    fi
    latency=$(python benchmarks/benchmark_latency.py --batch-size $batch --input-len $in --output-len $out --model /models/$model -tp $tp $eager |& grep "Avg latency:")
    echo "${model},${batch},${in},${out},${tp},${latency}"
}

function run_corectness()
{
    model=$1
    shift
    echo $model
    python /projects/llm_test.py --model $model $@ |& grep "Generated:"
}

echo "===Corectness==="

run_corectness /models/Meta-Llama-3.1-8B-Instruct
run_corectness /models/Meta-Llama-3.1-405B-Instruct -tp 8
run_corectness /models/mistral-ai-models/Mixtral-8x22B-v0.1/ -tp 4

echo "===Vision==="
VLLM_USE_TRITON_FLASH_ATTN=0 python /projects/llm_test.py --model /models/Llama-3.2-90B-Vision-Instruct --image-path /projects/image1.jpg --prompt "Describe this image" -tp 4 |& grep "Generated:"
VLLM_USE_TRITON_FLASH_ATTN=0 python /projects/llm_test.py --model /models/Llama-3.2-90B-Vision-Instruct-FP8-KV --image-path /projects/image1.jpg --prompt "Describe this image" -tp 4 |& grep "Generated:"

echo "===Performance==="

run_perf llama-2-70b-chat-hf 1 2048 128 8

batches="1 2 16 256"
ins="128 2048"
outs="1 128 2048"
models="Meta-Llama-3.1-8B-Instruct Meta-Llama-3.1-70B-Instruct"
for batch in $batches ; do
for in in $ins ; do
for out in $outs ; do
for model in $models ; do
run_perf $model $batch $in $out 8
done
done
done
done

run_perf Meta-Llama-3-8B-Instruct 1 2048 2048 8
run_perf Meta-Llama-3-8B-Instruct 64 256 256 8

run_perf Meta-Llama-3.1-405B-Instruct 1 128 2048 8
run_perf Meta-Llama-3.1-405B-Instruct 16 2048 2048 8

run_perf mistral-ai-models/Mixtral-8x22B-v0.1/ 16 1024 1024 8