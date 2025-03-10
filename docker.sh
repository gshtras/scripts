#!/bin/bash
#set -x
dry_run=0
interactive=0
grep_value=
command=
it=" -it"
USER=${USER:-$(whoami)}
image=${USER}_vllm
suffix=
name=${USER}_vllm
extra_args=
while : ; do
docker ps --format "{{.Names}}" | grep -sw $name$suffix &> /dev/null || break
suffix=$(( suffix + 1 ))
done
name=$name$suffix
entrypoint=
models_folder=/data/models
pull=1
while [[ $# -gt 0 ]] ; do
  i=$1
  case $i in
  -n|--name)
    name="$2"
    shift
  ;;
  --dry-run)
    dry_run=1
  ;;
  -g|--grep)
    grep_value="$2"
    shift
  ;;
  -i|--interactive)
    interactive=1
  ;;
  --noit)
    it=
  ;;
  -c|--command)
    command="$2"
    shift
  ;;
  --shmem)
    extra_args="--shm-size $2"
    shift
  ;;
  -m)
    models_folder="$2"
    shift
  ;;
  -e|--entrypoint)
    entrypoint="--entrypoint $2"
    shift
  ;;
  --nopull)
    pull=0
  ;;
  *)
    image=$1
  ;;
  esac
  shift
done

if [[ $interactive == 1 ]] ; then
    grep_arg=
    if [[ $grep_value != "" ]] ; then
      grep_arg=" | grep $grep_value"
    fi
    cmd="docker images | awk '{print \$1 \":\" \$2}' | grep -v none | grep -v REPOSITORY:TAG $grep_arg"
    images=$(eval $cmd)
    i=0
    for im in $images ; do
    echo "$i $im"
    i=$((i+1))
    done
    read -p "Select image: " selection
    images_arr=($images)
    image=${images_arr[$selection]}
    read -p "Container name: " name
fi
echo "Image: $image"

name_arg=
if [[ $name != "" ]] ; then
    name_arg=" --name $name -e CONTAINER_NAME=$name"
fi

if command -v nvidia-smi ; then
    IS_CUDA=1
    gpu_args="--runtime nvidia --gpus all"
elif command -v rocm-smi ; then
    IS_ROCM=1
    gpu_args="--device=/dev/kfd --device=/dev/dri --group-add video"
else
    echo "No GPU found"
    exit 1
fi

if [[ $pull == 1 ]] ; then
  docker pull $image
fi

full_cmd="docker run ${it} --rm ${gpu_args} -v /tmp/tmux-$(id -u):/tmp/tmux --mount type=bind,source=${HOME}/Projects,target=/projects --mount type=bind,source=${models_folder},target=/models --ulimit core=0:0 --ulimit memlock=-1:-1 $entrypoint $extra_args --cap-add=SYS_PTRACE $name_arg $image $command"
if [[ $dry_run != 1 ]] ; then
tmux rename-window "Docker:$name"
${full_cmd}
tmux setw automatic-rename on
echo "Finished docker image $image"
else
echo "Dry run: $full_cmd"
fi
