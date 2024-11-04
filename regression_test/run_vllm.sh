#!/bin/bash

set -e

function run()
{
    model=$1
    batch=$2
    in=$3
    out=$4
    tp=$5
    echo "Model: $model, Batch: $batch, Input: $in, Output: $out, TP: $tp"
    python benchmarks/benchmark_latency.py --batch-size $batch --input-len $in --output-len $out --model $model -tp $tp |& grep "Avg latency:"
}

function run_corectness()
{
    model=$1
    shift
    echo $model
    python /projects/llm_test.py --model $model $@ |& grep "Generated:"
}

echo "Corectness"

run_corectness /models/Meta-Llama-3.1-8B-Instruct
run_corectness /models/mistral-ai-models/Mixtral-8x22B-v0.1/ -tp 4

echo "Vision"
VLLM_USE_TRITON_FLASH_ATTN=0 python /projects/llm_test.py --model /models/Llama-3.2-90B-Vision-Instruct --image-path /projects/image1.jpg --prompt "Describe this image" -tp 4 |& grep "Generated:"

echo "Performance"

run /models/llama-2-70b-chat-hf 1 2048 128 8

run /models/Meta-Llama-3.1-8B-Instruct 1 2048 2048 8
run /models/Meta-Llama-3.1-8B-Instruct 64 256 256 8
run /models/Meta-Llama-3-8B-Instruct 1 2048 2048 8
run /models/Meta-Llama-3-8B-Instruct 64 256 256 8

run /models/Meta-Llama-3.1-70B-Instruct 1 128 2048 8
run /models/Meta-Llama-3.1-70B-Instruct 1 2048 128 8
run /models/Meta-Llama-3.1-70B-Instruct 64 256 256 8

run /models/mistral-ai-models/Mixtral-8x22B-v0.1/ 16 1024 1024 8