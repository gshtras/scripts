#!/bin/bash
#set -x
dry_run=0
interactive=0
grep_value=
cmd=
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
models_arg="--mount type=bind,source=/data/models,target=/models"
if [ -f ~/.models ] ; then
  models_arg=$(cat ~/.models | awk 'NF' | awk -F: '{printf "--mount type=bind,source=%s,target=%s ", $1, $2}')
fi
pull=1
no_gfx=0
no_rename=0
nethost=
shmem_arg="--shm-size 8G"
user_args=
while [[ $# -gt 0 ]] ; do
  i=$1
  case $i in
  -n|--name)
    name="$2"
    shift
  ;;
  --dry-run)
    dry_run=1
    no_rename=1
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
  -c|--cmd)
    cmd="$2"
    shift
  ;;
  --shmem)
    shmem_arg="--shm-size $2"
    shift
  ;;
  --no-shmem)
    shmem_arg=
  ;;
  -m)
    models_arg="--mount type=bind,source=${2},target=/models"
    shift
  ;;
  -e|--entrypoint)
    entrypoint="--entrypoint $2"
    shift
  ;;
  --nopull)
    pull=0
  ;;
  --env-file)
    extra_args="${extra_args} --env-file $2"
    shift
  ;;
  --no-gfx)
    no_gfx=1
  ;;
  --nethost)
    nethost="--network=host"
  ;;
  --no-rename)
    no_rename=1
  ;;
  -u)
    TMP_DIR=$(mktemp -d)
    user_args="--user $(id -u):$(id -g) -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v ${TMP_DIR}:/home/${USER} --workdir /home/${USER}"
  ;;
  --)
    shift
    cmd="$@"
    break
  ;;
  *)
    image=$1
  ;;
  esac
  shift
done
echo $command
extra_args="$extra_args $nethost $shmem_arg -e VLLM_DISABLE_COMPILE_CACHE=1 -e AITER_LOG_LEVEL=ERROR"

if [[ $interactive == 1 ]] ; then
    grep_arg=
    if [[ $grep_value != "" ]] ; then
      grep_arg=" | grep $grep_value"
    fi
    interactive_cmd="docker images | awk '{print \$1 \":\" \$2}' | grep -v none | grep -v REPOSITORY:TAG $grep_arg"
    images=$(eval $interactive_cmd)
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
    echo "Found CUDA"
    IS_CUDA=1
    gpu_args="--runtime nvidia --gpus all"
elif command -v rocm-smi ; then
    echo "Found ROCm"
    IS_ROCM=1
    gpu_args="--device=/dev/kfd --device=/dev/dri --group-add video"
    if [[ $no_gfx == 0 ]] ; then
        gpu_args="$gpu_args -e PYTORCH_ROCM_ARCH=$(/opt/rocm/lib/llvm/bin/amdgpu-arch | sort | uniq)"
    fi
elif [ -d /dev/dri ] ; then
    echo "No ROCm or CUDA installation found but /dev/dri exists, assuming AMD GPU"
    IS_ROCM=1
    gpu_args="--device=/dev/kfd --device=/dev/dri --group-add video"
else
    echo "No GPU found"
    exit 1
fi
if [[ $no_rename == 0 ]] ; then
  tmux rename-window "Docker:$name"
fi
if [[ $pull == 1 ]] ; then
  docker pull $image
fi

full_cmd="docker run ${it} --rm ${gpu_args} -v /tmp/tmux-$(id -u):/tmp/tmux --mount type=bind,source=${HOME}/Projects,target=/projects ${user_args} ${models_arg} --ulimit core=0:0 --ulimit memlock=-1:-1 $entrypoint $extra_args --cap-add=SYS_PTRACE $name_arg $image"
if [[ $dry_run != 1 ]] ; then
if [[ $cmd == "" ]] ; then
  ${full_cmd}
else
  ${full_cmd} bash -c "$cmd"
fi
if [[ $no_rename == 0 ]] ; then
  tmux setw automatic-rename on
fi
echo "Finished docker image $image"
else
echo "Dry run: $full_cmd"
fi
