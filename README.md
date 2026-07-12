# Confidential LLM inference benchmarking in CC

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
      - [Preparing the docker image for SGX](#preparing-the-docker-image-for-sgx)
      - [Running SGX docker image for Benchmark](#running-sgx-docker-image-for-benchmark)
    - [Quantizing models](#quantizing-models)
    - [Processing Results](#processing-results)
    - [Tracing](#tracing)

## Prerequisites

This reproduction runs the CPU TEE arms only, on two Azure Intel-based
confidential-computing VMs: `Standard_DC16s_v3` (Ice Lake, DCsv3 family) hosts
the baseline and SGX arms, and `Standard_DC16es_v6` (Emerald Rapids, DCesv6
family) hosts the TDX arm. Both run Ubuntu 24.04 LTS. Deploying those VMs is
covered in [AZURE.md](AZURE.md).

For SGX or TDX benchmarks, follow the respective sections on SGX or TDX
setup below. All benchmarks use Llama2 7B in bfloat16.

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

#### Preparing the docker image for SGX

`sgx_setup.sh` already builds this image (see [SGX Setup](#sgx-setup)). To rebuild it manually: it requires the `ipex-llm:2.2.0` base image to already exist (`sgx/Dockerfile.sgx` is `FROM ipex-llm:2.2.0`, not the 2.3.100 image used by the baseline/TDX track), then:

```sh
DOCKER_BUILDKIT=1 docker build -f sgx/Dockerfile.sgx -t sgx-ipex-llm:2.2.0 .
```

#### Running SGX docker image for Benchmark

Unlike the baseline/TDX arms, the SGX arm is not driven by `run.sh`: each configuration is one `gramine-sgx` invocation inside the `sgx-ipex-llm:2.2.0` container.

> [!IMPORTANT]
> **Run the baseline sweep on this machine first.** The Gramine enclave has no network access (DNS resolution fails inside it), so the model can only be loaded offline from the mounted `~/.cache` — which the baseline run populates. With an empty cache the SGX run dies with `Couldn't connect to huggingface.co ... couldn't find it in the cached files`. The failed HEAD requests to huggingface.co at startup are normal; with a populated cache transformers falls back to the local files.

One configuration, end to end, capturing the output into a file `run_parser.py` can read (the filename encodes every CSV column, so keep the pattern `sgx-<in>in-<out>out-<n>vCPU-1s-<bs>bs-7b-bf16.txt`):

```sh
d=results/$(date +"%F-%H-%M"); mkdir -p $d
docker run --rm --privileged --shm-size=2gb -v $HOME/.cache:/home/ubuntu/.cache sgx-ipex-llm:2.2.0 bash -c "\
  . ./miniconda3/bin/activate && conda activate py310 && \
  source ./llm/tools/env_activate.sh && cd ~/sgx && \
  numactl -m 0 -C 0-15 gramine-sgx LLM ~/llm/single_instance/run_generation.py \
    --ipex --token-latency --dtype bfloat16 -m meta-llama/Llama-2-7b-hf \
    --input-tokens 128 --max-new-tokens 128 \
    --num-iter 30 --num-warmup 10 --batch-size 1 --greedy --benchmark" \
  &> $d/sgx-128in-128out-16vCPU-1s-1bs-7b-bf16.txt
```

> [!IMPORTANT]
> Do not drop `--ipex --token-latency`. Unlike the deepspeed script, `run_generation.py` applies IPEX optimization only when `--ipex` is passed — without it the run measures vanilla-transformers inference, which is not comparable to the baseline/TDX arms (and uses far more memory: full attention matrices instead of IPEX's fused path). `--token-latency` (which requires `--ipex`) emits the per-token latency lists that `run_parser.py` and the latency analysis need.

Sweep the matrix by varying `--input-tokens` (128, 512, 2048) and `--batch-size` (1, 64), keeping the filename in sync. Match `run.sh`'s conventions: batch 1 uses `--greedy --num-warmup 10`; batch 64 drops `--greedy` and uses `--num-warmup 5`. Adjust `-C 0-15` to the machine's core list if not 16 vCPUs.

Notes:

- The first run takes several minutes before the first iteration prints: Gramine builds and measures a multi-GB enclave and loads the ~14 GB model through it. This is normal, not a hang.
- The `sgx.debug = true` warning appears on every run and is expected.
- Per-token latency will be visibly higher than the baseline on the same machine — that difference is the SGX overhead being measured.
- Batch-64 configurations can fail with `DefaultCPUAllocator: can't allocate memory` **inside the enclave**: `llm.manifest.template` sets `sgx.enclave_size = "64G"` (sized to the EPC), and at batch 64 the benchmark defaults to 4-beam search, so KV cache and prefill activations can exceed it. Raising `sgx.enclave_size` past the EPC and rebuilding the image makes Gramine rely on kernel EPC paging — the run may then complete, but with a heavy, measurable slowdown.

### Quantizing models
To quantize the models, follow `genQuantLLamaModels.sh`.

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
