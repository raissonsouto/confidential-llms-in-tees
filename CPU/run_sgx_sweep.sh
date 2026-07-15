#!/usr/bin/env bash
# Serial SGX benchmark sweep: input tokens {128,512,2048} x batch size {1,64}.
# Run detached: nohup bash run_sgx_sweep.sh > sweep.log 2>&1 &
set -u

d=results/$(date +"%F-%H-%M")
mkdir -p "$d"

lscpu > "$d/lscpu.out"
numactl --hardware > "$d/numactl-hw.out"

for in_tok in 128 512 2048; do
  for bs in 1 64; do
    # match run.sh conventions: greedy + warmup 10 for batch 1, no greedy + warmup 5 for batch 64
    if [ "$bs" -eq 1 ]; then greedy="--greedy"; warmup=10; else greedy=""; warmup=5; fi
    out="$d/sgx-${in_tok}in-128out-16vCPU-1s-${bs}bs-7b-bf16.txt"
    echo "$(date -Is) START input-tokens=$in_tok batch-size=$bs -> $out"
    docker run --rm --privileged --shm-size=2gb \
      -v "$HOME/.cache:/home/ubuntu/.cache" sgx-ipex-llm:2.2.0 bash -c "\
      . ./miniconda3/bin/activate && conda activate py310 && \
      source ./llm/tools/env_activate.sh && cd ~/sgx && \
      numactl -m 0 -C 0-15 gramine-sgx LLM ~/llm/single_instance/run_generation.py \
        --ipex --token-latency --dtype bfloat16 -m meta-llama/Llama-2-7b-hf \
        --input-tokens $in_tok --max-new-tokens 128 \
        --num-iter 30 --num-warmup $warmup --batch-size $bs $greedy --benchmark" \
      &> "$out"
    echo "$(date -Is) DONE  input-tokens=$in_tok batch-size=$bs (exit $?)"
  done
done

echo "$(date -Is) sweep finished: $d"
