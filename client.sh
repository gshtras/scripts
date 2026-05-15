#!/bin/bash

set -e
set -o pipefail
prompt="What is the best recipe for a margarita?   "
max_tokens=300
temperature=
batch_size=1
use_completions=0
port=8000
while [[ $# -gt 0 ]] ; do
  i=$1
  case $i in
  --prompt)
    prompt="$2"
    shift
  ;;
  --max-tokens)
    max_tokens="$2"
    shift
  ;;  
    --temperature)
    temperature=",\"temperature\": \"$2\""
    shift
  ;;
    --batch-size)
    batch_size="$2"
    shift
  ;;
  --port)
    port="$2"
    shift
  ;;
  --completions)
    use_completions=1
  ;;
  *)
    echo "Unknown option: $1"
    exit 1
  ;;
  esac
  shift
done

jq_cmd=
if [ $(which jq) ] ; then
    jq_cmd=$(which jq)
elif [ -f ~/jq ] ; then
    jq_cmd=~/jq
else
    curl -L -o ~/jq https://github.com/jqlang/jq/releases/download/jq-1.8.1/jq-linux-amd64 &> /dev/null
    chmod +x ~/jq
    jq_cmd=~/jq
fi
curl -s http://localhost:${port}/v1/models &> /dev/null || exit 1

model=$(curl -s http://localhost:${port}/v1/models | ${jq_cmd} ".data[0].root")
echo "Model: $model"

if [ $use_completions -eq 1 ] ; then
  if [ $batch_size -eq 1 ] ; then
    combined_prompt="\"${prompt}\""
  else
    combined_prompt="["
    for ((i=0;i<batch_size;i++)); do
        combined_prompt+="\"${prompt}\""
        if [ $i -lt $((batch_size-1)) ] ; then
            combined_prompt+=','
        fi
    done
    combined_prompt+="]"
  fi
  curl -s http://localhost:${port}/v1/completions -H "Content-Type: application/json" -d "{
      \"model\": ${model},
      \"prompt\": ${combined_prompt},
      \"max_tokens\": ${max_tokens}
      ${temperature}
  }" | ${jq_cmd}
    exit 0
fi

combined_prompt="["
for ((i=0;i<batch_size;i++)); do
    combined_prompt+="{\"role\": \"user\", \"content\": "\"${prompt}\""}"
    if [ $i -lt $((batch_size-1)) ] ; then
        combined_prompt+=','
    fi
done
combined_prompt+="]"

curl -s http://localhost:${port}/v1/chat/completions -H "Content-Type: application/json" -d "{
    \"model\": ${model},
    \"messages\": ${combined_prompt},
    \"max_tokens\": ${max_tokens}
    ${temperature}
}" | ${jq_cmd}
