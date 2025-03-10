#!/bin/bash

if [[ $(pwd) != *"vllm"* ]] ; then
    echo "Must be done in a vllm folder"
    exit 1
fi

rm -f vllm/core/block_manager.c \
      vllm/core/block_manager.html \
      vllm/core/scheduler.c \
      vllm/core/scheduler.html \
      vllm/engine/llm_engine.c \
      vllm/engine/llm_engine.html \
      vllm/engine/output_processor/single_step.c \
      vllm/engine/output_processor/single_step.html \
      vllm/engine/output_processor/stop_checker.c \
      vllm/engine/output_processor/stop_checker.html \
      vllm/model_executor/layers/sampler.c \
      vllm/model_executor/layers/sampler.html \
      vllm/outputs.c \
      vllm/outputs.html \
      vllm/sampling_params.c \
      vllm/sampling_params.html \
      vllm/sequence.c \
      vllm/sequence.html \
      vllm/transformers_utils/detokenizer.c \
      vllm/transformers_utils/detokenizer.html \
      vllm/utils.c \
      vllm/utils.html \
      vllm/outputs.cpython-312-x86_64-linux-gnu.so \
      vllm/sampling_params.cpython-312-x86_64-linux-gnu.so \
      vllm/sequence.cpython-312-x86_64-linux-gnu.so \
      vllm/utils.cpython-312-x86_64-linux-gnu.so
