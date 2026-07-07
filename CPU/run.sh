#!/bin/bash

# exit on failure
set -euxo pipefail

config=$1

# Disable AMX on every setup: cap the ATen/IPEX dispatch and the oneDNN JIT at
# the AVX512-BF16 ISA level so results are comparable across CPU generations
# regardless of whether the host exposes AMX.
config_no_amx='export ATEN_CPU_CAPABILITY=avx512_bf16 ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16'

config_num_iter=30
config_num_warmup=10
config_out_token=128
config_in_token=1024
config_procs=120
config_socket=$(( config_procs / 2 )) 

# per date folder
date=$(date +"%F-%H-%M")
directory=results/$date
mkdir -p $directory

echo "storing results in $directory"
echo "$1 stored in $directory" >> experiment.log

{
    lscpu &> $directory/lscpu.out
    lshw &> $directory/lshw.out
    numactl --hardware &> $directory/numactl-hw.out

    # initialize variables with different values
    for vCPUs in '1' '1-2' '1-4' '1-8' '1-16' '1-32' '1-48' '0-59'; do # if you want to use all cores available to the system just leave empty ''; if you want to use cores accross sockets, you can use `--num_accelerators 2` in the main command
        for batch_size in 1 64; do
            for in_token in 128 512 2048; do
                for out_token in 128; do
                    for quant in ''; do # bfloat16 only; INT8 quantization removed
                        for model in 'meta-llama/Llama-2-7b-hf'; do
                            num_iter=$config_num_iter
                            num_warmup=$config_num_warmup
                            # cmp output name
                            name=$directory/$1
                            name=$name-${in_token}in
                            name=$name-${out_token}out
                            if [[ -n ${vCPUs//[[:space:]]/} ]]; then
                                vCPUs_num=$(awk -F- '{print (NF==1)?$1:($2-$1+1)}' <<< "$vCPUs")
                                vCPUs="--bind_core_list $vCPUs"
                            else
                                vCPUs_num=$config_procs
                            fi
                            name=$name-${vCPUs_num}vCPU
                            if (( vCPUs_num > config_socket )); then
                                numa='2s'
                            else
                                numa='1s'
                            fi
                            name=$name-$numa
                            name=$name-${batch_size}bs
                            name=$name-7b
                            name=$name-bf16
                            # set greedy if single batch
                            greedy=''
                            if [[ "$batch_size" -eq 1 ]]; then
                                greedy='--greedy'
                            else
                                num_warmup=$(( $num_warmup/2 ))
                            fi

                            # safety
                            if [[ "$num_warmup" -le 1 ]]; then
                                num_warmup=2
                            fi

                            cmd=(docker run --rm --privileged --shm-size="2gb" -v $HOME/.cache:/home/ubuntu/.cache ipex-llm:2.3.100 bash -c \
                                "$config_no_amx && cd llm && source ../miniforge3/bin/activate && conda activate py310 && source tools/env_activate.sh && sudo chown -R 1000:1000 ~/.cache && deepspeed --bind_cores_to_rank $vCPUs distributed/run_generation_with_deepspeed.py --deployment-mode --profile --benchmark -m $model $quant --ipex --dtype bfloat16 --batch-size $batch_size --num-iter $num_iter --num-warmup $num_warmup --max-new-tokens $out_token --input-tokens $in_token --token-latency $greedy" )

                            # log cmd
                            echo "${cmd[@]}" > $name.txt

                            # run cmd
                            "${cmd[@]}" &>> $name.txt

                            # Finished run
                            echo "Finished"
                            exit 0
                        done
                    done
                done
            done
        done
    done
# store run log
} &> $directory/run.out


