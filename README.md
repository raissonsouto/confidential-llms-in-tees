# Confidential LLM inference benchmarking in CC

> This is a fork of [spcl/confidential-llms-in-tees](https://github.com/spcl/confidential-llms-in-tees),
> the reference implementation for the paper
> ["Confidential LLM Inference: Performance and Cost Across CPU and GPU TEEs" (arXiv:2509.18886)](https://arxiv.org/abs/2509.18886).
> This fork reproduces the CPU TEE arms (baseline, SGX, TDX) of that paper on
> a reduced sweep, and the GPU TEE arm (H100 vs H100 + Intel TDX) on the same
> reduced grid; see [Prerequisites](#prerequisites) below for the scope.

Repository to include scripts to run inference benchmarks in CC environments.

## Table of contents

- [Confidential LLM inference benchmarking in CC](#confidential-llm-inference-benchmarking-in-cc)
  - [Table of contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
    - [Hugging Face access token](#hugging-face-access-token)
  - [CPUs](#cpus)
    - [Common Setup](#common-setup)
    - [SGX Setup](#sgx-setup)
    - [Running baseline experiments](#running-baseline-experiments)
    - [Running TDX experiments](#running-tdx-experiments)
    - [Running SGX experiments](#running-sgx-experiments)
    - [Processing Results](#processing-results)
      - [Generating figures for this reproduction's dataset](#generating-figures-for-this-reproductions-dataset)
      - [Statistical analysis (medians, bootstrap CIs, Mann-Whitney U)](#statistical-analysis-medians-bootstrap-cis-mann-whitney-u)
    - [Tracing](#tracing)
  - [GPUs](#gpus)
    - [Pre-flight check](#pre-flight-check)
    - [Provisioning the VMs](#provisioning-the-vms)
    - [VM setup](#vm-setup)
    - [Smoke test](#smoke-test)
    - [Running the GPU sweep](#running-the-gpu-sweep)
    - [Collecting results](#collecting-results)
    - [Processing GPU results](#processing-gpu-results)
    - [GPU caveats](#gpu-caveats)

## Prerequisites

This reproduction covers the **CPU TEE arms** and the **GPU TEE arm** on the
same reduced grid: batch size 1 and 64, input length 128, 512 and 2048, 128
output tokens, Llama-2-7B in bfloat16, with 10 warmup and 30 measured
iterations per configuration.

The CPU arms run on two Azure Intel-based confidential-computing VMs:
`Standard_DC16s_v3` (Ice Lake, DCsv3 family) hosts the baseline and SGX arms,
and `Standard_DC16es_v6` (Emerald Rapids, DCesv6 family) hosts the TDX arm.
Both run Ubuntu 24.04 LTS. Deploying those VMs is covered in
[AZURE.md](AZURE.md).

The GPU arms run on two Google Cloud `a3-highgpu-1g` instances (1× NVIDIA H100
80 GB, Ubuntu 22.04 LTS): one plain, one with Intel TDX and GPU
confidential-computing mode. Deploying those is covered in
[GOOGLE_CLOUD.md](GOOGLE_CLOUD.md). Azure is used for the CPU arms and Google
Cloud for the GPU arms because Azure does not offer a confidential H100 in a
form comparable to the paper's, whereas Google Cloud does.

For SGX or TDX benchmarks, follow the respective sections on SGX or TDX
setup below; for the GPU arms, see [GPUs](#gpus). All benchmarks use Llama2 7B
in bfloat16.

### Hugging Face access token

The benchmarks download gated models ([`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf) — the one used by `run.sh`, [`Llama-2-13b-hf`](https://huggingface.co/meta-llama/Llama-2-13b-hf), [`Llama-2-70b-hf`](https://huggingface.co/meta-llama/Llama-2-70b-hf), and any Llama-3 variants you enable in `run.sh`). A token alone is not enough to pull these weights; you need both:

1. **Repo access**: visit each gated model's page linked above while logged into the account that owns the token, and accept Meta's license/usage agreement. Access is granted per model, so repeat this for every Llama variant you plan to run. Without this, `huggingface-cli login` succeeds but the download fails with a 403 error.

2. **Token permissions**: create the token at `https://huggingface.co/settings/tokens`.
   - Classic tokens: the `read` role is sufficient (do not use `write`/`fine-grained-write`).
   - Fine-grained tokens: enable "Read access to contents of all public gated repos you can access" under the "Repositories" permissions, or scope it explicitly to the model repos above.

## CPUs

### Common Setup

To setup the host for running experiments, please first initalize the repository, by cloning it and applying appropriate patches.

```sh
git clone https://github.com/raissonsouto/confidential-llms-in-tees.git
cd confidential-llms-in-tees

git checkout develop
git submodule sync
git submodule update --init --recursive

cd CPU/tdx
git apply ../tdx.patch

cd ../intel-extension-for-pytorch
git apply ../ipex.patch

cd ..
```

Then run the host setup script which will setup hugging face, create Docker, and build the necessary image. It reads `HUGGINGFACE_TOKEN` from `.env` at the repo root (`cp config.env .env` and fill it in), or you can pass the token inline:

```sh
./host_setup.sh                            # token from .env
HUGGINGFACE_TOKEN=<token> ./host_setup.sh  # or inline
```

See [Hugging Face access token](#hugging-face-access-token) in Prerequisites for what permissions this token needs. Relogin to apply changes in groups. Finally, compile the docker container — the build context must be the `intel-extension-for-pytorch` directory itself, since its Dockerfile copies the context to `./intel-extension-for-pytorch` inside the image:

```sh
cd intel-extension-for-pytorch

# ~10min
DOCKER_BUILDKIT=1 docker build -f examples/cpu/inference/python/llm/Dockerfile -t ipex-llm:2.3.100 .
cd ..
```

### SGX Setup

Gramine lives **inside** the SGX docker image (`sgx/Dockerfile.sgx` builds it from the `gramine/` sources), not on the host. The host only needs docker and the SGX devices (`ls /dev/sgx*`).

`sgx_setup.sh` builds everything: the `ipex-llm:2.2.0` base image (the SGX track runs on ipex 2.2 with `ipex-2.2.patch`, unlike the baseline/TDX track on 2.3), then the graminized `sgx-ipex-llm:2.2.0`, and finally quantizes models to INT8 — **skip that part for the bf16-only experiment** (the 70B download alone needs over 100 GB). From `CPU/`:

```sh
sed -i 's/^docker run/# docker run/' sgx_setup.sh   # skip INT8 quantization
./sgx_setup.sh                                       # ~10-15 min
git checkout -- sgx_setup.sh
```

Verify with the Gramine hello world inside the image:

```sh
docker run --rm --privileged sgx-ipex-llm:2.2.0 bash -c "cd gramine/CI-Examples/helloworld && make SGX=1 && gramine-sgx helloworld"
```

It should print `Hello, world` after a `sgx.debug = true` warning (expected: debug manifests, fine for benchmarking). For other Gramine errors, see [its documentation](https://gramine.readthedocs.io/en/stable/).

### Running baseline experiments

```sh
nohup ./run.sh baseline &
```

This will generate a folder under `results/` with the current date and time and add an entry into the experiment log. All generated files will have the form `baseline-system-in_size-out_size-vCPUs-numa-batch_size-model-data_type.txt`.

### Running TDX experiments
SSH to the TDX machine (the Azure `DC16es_v6` VM):
```sh
ssh azureuser@<tdx-vm-ip>
```

Run the experiments via:

```sh
nohup ./run.sh tdx &
```

### Running SGX experiments

Unlike the baseline/TDX arms, the SGX arm is not driven by `run.sh`: each configuration is one `gramine-sgx` invocation inside the `sgx-ipex-llm:2.2.0` container built by `sgx_setup.sh` (see [SGX Setup](#sgx-setup)). `run_sgx_sweep.sh` runs the full matrix — input tokens 128/512/2048 × batch size 1/64, 128 output tokens — serially, following `run.sh`'s conventions. From `CPU/`:

> [!IMPORTANT]
> **Run the baseline sweep on this machine first.** The Gramine enclave has no network access (DNS resolution fails inside it), so the model can only be loaded offline from the mounted `~/.cache` — which the baseline run populates. With an empty cache the SGX run dies with `Couldn't connect to huggingface.co ... couldn't find it in the cached files`. The failed HEAD requests to huggingface.co at startup are normal; with a populated cache transformers falls back to the local files.

```sh
nohup bash run_sgx_sweep.sh > sweep.log 2>&1 &
```

Results and hardware snapshots are written to a timestamped folder under `results/`, with the same file naming `run_parser.py` parses; per-configuration progress and exit codes go to `sweep.log`, and the `nohup` launch survives SSH disconnects. Adjust `-C 0-15` in the script if the machine does not have 16 vCPUs.

> [!IMPORTANT]
> Do not drop `--ipex --token-latency`. Unlike the deepspeed script, `run_generation.py` applies IPEX optimization only when `--ipex` is passed — without it the run measures vanilla-transformers inference, which is not comparable to the baseline/TDX arms (and uses far more memory: full attention matrices instead of IPEX's fused path). `--token-latency` (which requires `--ipex`) emits the per-token latency lists that `run_parser.py` and the latency analysis need.

Notes:

- The first run takes several minutes before the first iteration prints: Gramine builds and measures a multi-GB enclave and loads the ~14 GB model through it. This is normal, not a hang.
- The `sgx.debug = true` warning appears on every run and is expected.
- Per-token latency will be visibly higher than the baseline on the same machine — that difference is the SGX overhead being measured.
- Batch-64 configurations can fail with `DefaultCPUAllocator: can't allocate memory` **inside the enclave**: `llm.manifest.template` sets `sgx.enclave_size = "64G"` (sized to the EPC), and at batch 64 the benchmark defaults to 4-beam search, so KV cache and prefill activations can exceed it. Raising `sgx.enclave_size` past the EPC and rebuilding the image makes Gramine rely on kernel EPC paging — the run may then complete, but with a heavy, measurable slowdown.

### Processing Results

`processing/run_parser.py` gathers all iteration and token latencies from each
experiment and places them into a csv file. Its glob is recursive, so it accepts
either the parent `results` directory or a single `results/<date>-<time>` folder.
Run it from `CPU/`:

```sh
python3 processing/run_parser.py results
```

The output is written to `./results.csv` in the current directory and is
**overwritten on every run**, so copy it elsewhere before re-parsing. The CSV
can then be plotted with the helper functions in `processing/`.

#### Generating figures for this reproduction's dataset

The scripts in `processing/` were originally written for the full paper
sweep (multiple vCPU counts, dual-socket NUMA, AMX on/off, 7B/13B/70B,
bf16/int8). This reproduction only covers a single vCPU count (16), single
socket, Llama-2-7B, bf16, batch size 1/64, input 128/512/2048 — so five of
the original scripts were adapted to this reduced shape (`AMX*.py`,
`price.py`, `model_scaling_double_socket.py`, `model_scaling_70B.py`, and
`traces_parser.py` were left untouched: they need dimensions — AMX
on/off, dual socket, 70B — that were never measured here). All of them read
`results/results.csv` (produced by `run_parser.py` above) and are run from
`CPU/processing/`:

```sh
cd CPU/processing
python3 model_scaling_single_socket.py ../../results/results.csv   # throughput, baseline vs SGX vs TDX, by input size
python3 batch_size_scaling.py         ../../results/results.csv   # throughput vs batch size (1 vs 64)
python3 vCPUs_batch_size.py           ../../results/results.csv   # throughput + estimated cost, faceted by batch size
python3 vCPUs_input.py                ../../results/results.csv   # throughput + estimated cost, faceted by input size
python3 price_azure.py                                            # Azure DCsv3/DCesv6 on-demand price per vCPU (live API, no args)
```

Each writes one PNG directly to `results/` (`overall_single_socket.png`,
`batch_scaling_combined.png`, `vCPUs_GPU_EMR_batches.png`,
`vCPUs_GPU_EMR_inputs.png`, `azure_price.png`). `price_azure.py` needs
outbound network access (it queries `prices.azure.com` live instead of using
hardcoded prices); the other four are offline and only need `results.csv`.

#### Statistical analysis (medians, bootstrap CIs, Mann-Whitney U)

`processing/overhead_stats.py` reproduces the numbers in the paper's overhead
table: per-configuration median throughput/latency with 95% bootstrap
confidence intervals (10,000 resamples), and a two-sided Mann-Whitney U test
of each TEE against the VM baseline at the same batch-size/input-length cell.
It needs `results.csv` (from `run_parser.py` above) and the raw log
directory (`results/`, with the `baseline/`, `sgx/`, `tdx/` subfolders) to
recover the per-cell prefill cost used to reconstruct next-token latency:

```sh
pip install pandas numpy scipy
python3 CPU/processing/overhead_stats.py results/results.csv results
```

Run from the repository root. The bootstrap uses a fixed RNG seed, so the
output is deterministic across runs. No Azure access is needed: this script
only reads the already-published `results/` data, so it reproduces the
paper's confidence intervals and $p$-values without re-running any benchmark.

### Tracing
To obtain traces, start the Docker container:
```
docker run --rm --privileged --shm-size=2gb -it -v /home/mchrapek/.cache:/home/ubuntu/.cache ipex-llm:2.3.100 bash 
```
Inside run the inference command with `--profile`, e.g.:
```
export ATEN_CPU_CAPABILITY=avx512 ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16 LIBXSMM_TARGET=cpx && cd llm && source ../miniforge3/bin/activate && conda activate py310 && source tools/env_activate.sh && sudo chown -R 1000:1000 ~/.cache && deepspeed --bind_cores_to_rank --num_accelerators 1 --bind_core_list 0-59 distributed/run_generation_with_deepspeed.py --deployment-mode --benchmark -m meta-llama/Llama-2-7b-hf --ipex --dtype bfloat16 --batch-size 64 --num-iter 15 --num-warmup 5 --max-new-tokens 128 --input-tokens 128 --token-latency --greedy --profile
```
This will generate log files which can be processed and plotted by `traces_parser.py`. It accepts two files with traces that correspond to two compared systems.

## GPUs

The GPU arm compares an NVIDIA H100 against the same H100 running under Intel
TDX with GPU confidential-computing mode enabled, on the same reduced grid as
the CPU arms: **batch size 1 and 64 × input length 128, 512 and 2048**, 128
output tokens, Llama-2-7B in bfloat16, 10 warmups and 30 measured iterations
per configuration. Inference is served by [vLLM](https://github.com/vllm-project/vllm),
pinned to `v0.9.2`, and driven through its `benchmarks/benchmark_latency.py`.

Both VMs are Google Cloud `a3-highgpu-1g` instances — see
[GOOGLE_CLOUD.md](GOOGLE_CLOUD.md) for provisioning. Two instances are needed
because GPU CC mode is fixed at instance creation and cannot be toggled from
inside the guest, so unlike the Azure SGX VM, one machine cannot host both arms.

> [!IMPORTANT]
> `a3-highgpu-1g` is offered **only** as a Spot (or flex-start) instance, and
> Confidential VM with TDX cannot use reservations. Both arms are therefore
> preemptible, and at roughly $10/hour each. Work through the pre-flight check
> and the smoke test before starting the full sweep — they exist to move
> failures off the clock.

### Pre-flight check

Verifies the whole environment before anything bills: required binaries,
gcloud authentication and project, Compute Engine API, the IAM permissions
needed to create and delete an instance, H100 spot quota in all three
supported regions, machine-type and image availability, and — the one most
likely to bite — that your Hugging Face token actually has access to the gated
Llama-2 repo.

```sh
cp config.env .env     # fill in HUGGINGFACE_TOKEN and GCP_PROJECT
cd GPU
./preflight.sh
```

It creates nothing and exits non-zero on the first problem. Don't provision
until it exits 0.

### Provisioning the VMs

Follow [GOOGLE_CLOUD.md](GOOGLE_CLOUD.md#baseline-gpu-vm). In short, from the
repo root with `.env` loaded:

```sh
source .env
gcloud compute instances create $CGPU_VM_NAME \
  --zone=$GCP_ZONE --machine-type=$GPU_MACHINE_TYPE \
  --confidential-compute-type=TDX \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --image-project=$GPU_IMAGE_PROJECT --image-family=$GPU_IMAGE_FAMILY \
  --boot-disk-size=$GPU_BOOT_DISK_SIZE --boot-disk-type=pd-balanced
```

Drop `--confidential-compute-type=TDX` for the baseline VM. Everything else
about the two instances is identical on purpose, so the TEE is the only
variable.

### VM setup

Copy the repo across and run the setup script on the instance. It installs the
NVIDIA driver (580+, required for CC mode), enables the LKCA and persistence
settings CC mode needs, installs vLLM, downloads the weights, and captures a
hardware snapshot:

```sh
gcloud compute scp --recurse --zone=$GCP_ZONE GPU .env \
  $CGPU_VM_NAME:~/confidential-llms-in-tees/
gcloud compute ssh $CGPU_VM_NAME --zone=$GCP_ZONE

cd ~/confidential-llms-in-tees/GPU
./gcp_vm_setup.sh cgpu     # reboots once; reconnect and re-run to finish
```

Pass `gpu` instead of `cgpu` on the baseline VM — it then skips the CC-mode
changes and leaves the machine stock, so it stays a clean control.

> [!IMPORTANT]
> On the confidential VM, confirm the GPU really is in CC mode before
> measuring anything:
> ```sh
> sudo nvidia-smi conf-compute -f     # must print: CC status: ON
> ```
> A confidential VM whose GPU came up with `CC status: OFF` yields a second
> baseline run under a confidential label, and the "overhead" you report is
> noise around zero. `gcp_vm_setup.sh` refuses to continue in that case.

### Smoke test

One configuration — **batch 1, input 128** — on the confidential VM. This is the
smallest cell in the grid, and the smoke test's job is to prove the pipeline
works end to end (driver, CC mode, vLLM, weights, JSON output) for the least
GPU time possible:

```sh
source ~/.venv/bin/activate
./benchmark_vllm.sh cgpu --smoke
```

Check that the resulting `latency_in128_bs1.json` has 30 entries in
`latencies`, and note the `GPU KV cache size` / `Maximum concurrency` lines
the script extracts:

```sh
jq '.latencies | length' results_cgpu_*/latency_in128_bs1.json
cat results_cgpu_*/latency_in128_bs1.log.kv
```

The heaviest cell, batch 64 at input 2048, is deliberately *not* the smoke
test: whether it fits in 80 GB is one of the things the sweep is measuring
(see [GPU caveats](#gpu-caveats)), so it belongs in the run rather than in the
gate that precedes it.

### Running the GPU sweep

Same VM, all six configurations. The smoke test's JSON is already present, so
it is skipped rather than re-run:

```sh
source ~/.venv/bin/activate
RESULTS_DIR=results_cgpu_<timestamp> nohup ./benchmark_vllm.sh cgpu > sweep-cgpu.log 2>&1 &
```

Then repeat on the baseline VM with `./benchmark_vllm.sh gpu`.

`nohup` keeps the sweep alive across SSH drops. The sweep is also **resumable**:
each configuration whose `.json` already exists is skipped, so a Spot
preemption costs the configuration in flight and nothing else. Restart the
stopped instance and re-run the same command with the same `RESULTS_DIR`.

### Collecting results

Copy each arm's results down as soon as it finishes — a deleted instance takes
its boot disk with it:

```sh
gcloud compute scp --recurse --zone=$GCP_ZONE \
  $CGPU_VM_NAME:~/confidential-llms-in-tees/GPU/results_cgpu_\* ./results/cgpu/
gcloud compute scp --recurse --zone=$GCP_ZONE \
  $CGPU_VM_NAME:~/confidential-llms-in-tees/GPU/hwinfo-cgpu ./results/cgpu/
```

Then delete the instances (see
[Cleaning up](GOOGLE_CLOUD.md#cleaning-up)).

### Processing GPU results

Unlike the CPU track, GPU results are **not** folded into
`results/results.csv` — vLLM emits its own per-configuration JSON, and the two
tracks measure different systems on different clouds. They are processed by
their own scripts, run from `GPU/`:

```sh
cd GPU
python3 parse.py      ../results/gpu ../results/cgpu   # latency, throughput, $/Mtok, overhead
python3 plot_GPUs.py  ../results/gpu ../results/cgpu   # throughput comparison figure
```

`parse.py` prices both arms at the `a3-highgpu-1g` spot rate; override it for a
specific run with `GPU_COST_PER_HOUR=<usd> python3 parse.py ...`. `plot_GPUs.py`
writes `results/gpu_throughput_comparison.png`: throughput vs batch size at a
fixed input length, and vs input length at a fixed batch size, with each
confidential bar annotated with its overhead against the baseline.

### GPU caveats

- **Batch 64 at input 2048 is expected to be tight, and its outcome is a
  result.** Llama-2-7B uses multi-head attention, so its KV cache costs about
  0.5 MB per token: batch 64 × (2048 + 128) tokens needs roughly 68 GB, on top
  of about 13.5 GB of weights, against an 80 GB H100. That cell may OOM, or may
  run with the batch split across scheduler waves. Either outcome is a finding
  about the memory ceiling of confidential inference at this shape, and is
  reported as such — the CPU track records its OOM cells the same way (see
  [AZURE.md](AZURE.md#virtual-machines-used)). The sweep runs at
  `--gpu-memory-utilization 0.95 --max-model-len 2176` and records vLLM's own
  `GPU KV cache size` and `Maximum concurrency` lines per configuration in
  `<log>.kv`, so what actually happened is evidenced rather than inferred.
  Override with `GPU_MEM_UTIL` and `MAX_MODEL_LEN` to measure a different
  point.
- **Both arms are Spot instances.** This is forced by the platform, not chosen.
  It makes the two arms symmetric, but it also means neither arm has a
  guaranteed-uninterrupted host, and run-to-run variance may be higher than on
  dedicated hardware.
- **Ubuntu 22.04, not 24.04.** The GPU arms run 22.04 because that is the only
  Ubuntu image Google supports for Confidential VM with GPU; the CPU arms run
  24.04. Compare TEE-vs-baseline ratios within a track rather than absolute
  numbers across tracks.
