#!/usr/bin/env bash
#
# GPU inference sweep, driven through vLLM's benchmark_latency.py.
#
#     ./benchmark_vllm.sh gpu             # baseline H100
#     ./benchmark_vllm.sh cgpu            # H100 with TDX + GPU CC mode
#     ./benchmark_vllm.sh cgpu --smoke    # only batch 64 / input 2048
#
# The grid matches the CPU reproduction (CPU/run.sh) so the two tracks are
# directly comparable: batch 1/64 x input 128/512/2048, 128 output tokens,
# Llama-2-7B in bfloat16.
#
# Results go to results_<system>_<timestamp>/latency_in<IN>_bs<BS>.{log,json}.
# Set RESULTS_DIR to resume into an existing folder.

set -uo pipefail

SYSTEM="${1:-}"
if [ "$SYSTEM" != "gpu" ] && [ "$SYSTEM" != "cgpu" ]; then
    echo "usage: $0 {gpu|cgpu} [--smoke]" >&2
    exit 1
fi
SMOKE=0
[ "${2:-}" = "--smoke" ] && SMOKE=1

# Model and benchmark settings
MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
OUTPUT_LEN=128
DTYPE="bfloat16"
NUM_WARMUP="${NUM_WARMUP:-10}"
NUM_ITERS="${NUM_ITERS:-30}"

# vLLM ships benchmark_latency.py in its git tree, not in the pip wheel.
VLLM_REPO="${VLLM_REPO:-$HOME/vllm}"
BENCH="$VLLM_REPO/benchmarks/benchmark_latency.py"
if [ ! -f "$BENCH" ]; then
    echo "ERROR: $BENCH not found. Set VLLM_REPO to your vLLM checkout." >&2
    echo "gcp_vm_setup.sh clones it to ~/vllm." >&2
    exit 1
fi

# Llama-2-7B uses MHA, so its KV cache is ~0.5 MB/token: batch 64 x (2048+128)
# tokens needs ~68 GB on top of ~13.5 GB of weights, which does not fit an 80 GB
# H100 at vLLM's default utilisation of 0.9. Raising it to 0.95 and capping the
# context at exactly what the sweep needs buys back enough KV cache to matter.
# If the batch still has to be split across scheduler waves, the per-config log
# records it (see the "KV cache" grep below) so the cell can be reported
# honestly rather than as a clean batch of 64.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2176}"

if [ "$SMOKE" = "1" ]; then
    INPUT_LENS=(2048)
    BATCHES=(64)
else
    INPUT_LENS=(128 512 2048)
    BATCHES=(1 64)
fi

# Timestamped directory for results, or an existing one to resume into.
TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
RESULTS_DIR="${RESULTS_DIR:-results_${SYSTEM}_${TIMESTAMP}}"
mkdir -p "${RESULTS_DIR}"

echo "system=${SYSTEM} model=${MODEL} warmup=${NUM_WARMUP} iters=${NUM_ITERS}"
echo "results -> ${RESULTS_DIR}"
echo

FAILED=0
for INPUT_LEN in "${INPUT_LENS[@]}"; do
  for BATCH in "${BATCHES[@]}"; do
    LOG_FILE="${RESULTS_DIR}/latency_in${INPUT_LEN}_bs${BATCH}.log"
    JSON_FILE="${RESULTS_DIR}/latency_in${INPUT_LEN}_bs${BATCH}.json"

    # Resume support: a Spot preemption mid-sweep should cost one configuration,
    # not the whole run. a3-highgpu-1g is Spot-only, so this is not hypothetical.
    if [ -s "${JSON_FILE}" ]; then
      echo "===== SKIP input=${INPUT_LEN} batch=${BATCH} (already have $(basename "${JSON_FILE}")) ====="
      continue
    fi

    echo "===== $(date '+%F %T') Input length = ${INPUT_LEN}, Batch size = ${BATCH} =====" | tee "${LOG_FILE}"

    VLLM_USE_V1=0 python "${BENCH}" \
      --model "${MODEL}" \
      --input-len "${INPUT_LEN}" \
      --output-len "${OUTPUT_LEN}" \
      --batch-size "${BATCH}" \
      --dtype "${DTYPE}" \
      --gpu-memory-utilization "${GPU_MEM_UTIL}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --output-json "${JSON_FILE}" \
      --num-iters-warmup "${NUM_WARMUP}" \
      --num-iters "${NUM_ITERS}" \
      2>&1 | tee -a "${LOG_FILE}"
    STATUS=${PIPESTATUS[0]}

    if [ "${STATUS}" -ne 0 ]; then
      echo "FAILED input=${INPUT_LEN} batch=${BATCH} (exit ${STATUS})" | tee -a "${LOG_FILE}"
      FAILED=$((FAILED + 1))
    fi

    # Pull vLLM's own account of how much KV cache it got and how many of the
    # requested sequences actually fit concurrently. This is the evidence for
    # (or against) the batch-64 caveat above.
    grep -E 'KV cache size|Maximum concurrency' "${LOG_FILE}" | tee -a "${LOG_FILE}.kv" || true

    echo "Results -> log: ${LOG_FILE}, json: ${JSON_FILE}"
    echo
  done
done

echo "Sweep finished with ${FAILED} failed configuration(s)."
echo "Results in ${RESULTS_DIR}"
[ "${FAILED}" -eq 0 ]
