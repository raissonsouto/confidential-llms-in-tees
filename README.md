# Confidential LLM inference benchmarking in CC

Repository to include scripts to run inference benchmarks in CC environments. Deploying the Azure VMs is covered in [AZURE.md](AZURE.md).

## Table of contents

- [Confidential LLM inference benchmarking in CC](#confidential-llm-inference-benchmarking-in-cc)
  - [Table of contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [CPUs](#cpus)
    - [Common Setup](#common-setup)
      - [Hugging Face access token](#hugging-face-access-token)
    - [SGX Setup](#sgx-setup)
    - [TDX Setup](#tdx-setup)
      - [Prepare a TDX VM image](#prepare-a-tdx-vm-image)
      - [Copy the repository to the VM](#copy-the-repository-to-the-vm)
      - [Enable hugepages](#enable-hugepages)
    - [Running baseline experiments](#running-baseline-experiments)
    - [Running TDX experiments](#running-tdx-experiments)
    - [Running SGX experiments](#running-sgx-experiments)
      - [Preparing the docker image for SGX](#preparing-the-docker-image-for-sgx)
      - [Running SGX docker image for Benchmark](#running-sgx-docker-image-for-benchmark)
    - [Quantizing models](#quantizing-models)
    - [Processing Results](#processing-results)
    - [Tracing](#tracing)
  - [GPU](#gpu)
  - [RAG](#rag)

## Prerequisites

In our work we run on SPR or EMR Intel Xeon (generation 4 or older) CPUs and H100 GPUs. We used Ubuntu 24.04 as the host OS. Later Ubuntu versions should also work.

For benchmarks with SGX or TDX, please follow the respective sections on SGX or TDX setup.
For GPU benchmarks, follow the GPU section.
Finally, for RAG benchmarks, see the corresponding section. Note RAG currently only operates on CPUs.

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

See [Hugging Face access token](#hugging-face-access-token) below for what permissions this token needs. Relogin to apply changes in groups. Finally, compile the docker container — the build context must be the `intel-extension-for-pytorch` directory itself, since its Dockerfile copies the context to `./intel-extension-for-pytorch` inside the image:

```sh
cd intel-extension-for-pytorch

# ~10min
DOCKER_BUILDKIT=1 docker build -f examples/cpu/inference/python/llm/Dockerfile -t ipex-llm:2.3.100 .
cd ..
```

#### Hugging Face access token

The benchmarks download gated models ([`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf) — the one used by `run.sh`, [`Llama-2-13b-hf`](https://huggingface.co/meta-llama/Llama-2-13b-hf), [`Llama-2-70b-hf`](https://huggingface.co/meta-llama/Llama-2-70b-hf), and any Llama-3 variants you enable in `run.sh`). A token alone is not enough to pull these weights; you need both:

1. **Repo access**: visit each gated model's page linked above while logged into the account that owns the token, and accept Meta's license/usage agreement. Access is granted per model, so repeat this for every Llama variant you plan to run. Without this, `huggingface-cli login` succeeds but the download fails with a 403 error.

2. **Token permissions**: create the token at `https://huggingface.co/settings/tokens`.
   - Classic tokens: the `read` role is sufficient (do not use `write`/`fine-grained-write`).
   - Fine-grained tokens: enable "Read access to contents of all public gated repos you can access" under the "Repositories" permissions, or scope it explicitly to the model repos above.

### SGX Setup

Gramine is **not** installed on the host: `sgx/Dockerfile.sgx` builds and installs it from the `gramine/` sources inside the SGX docker image, so `gramine-sgx` only exists inside that image (running it on the host gives `command not found`).

`sgx_setup.sh` automates the SGX preparation: it checks out ipex `release/2.2` and builds the `ipex-llm:2.2.0` base image (the SGX track runs on ipex 2.2, unlike the baseline/TDX track), builds the graminized `sgx-ipex-llm:2.2.0` image from `sgx/Dockerfile.sgx`, restores ipex to `release/2.3`, and finally quantizes the 7B/13B/70B models to INT8. **For the bf16-only experiment, skip the quantization part** (the 70B download alone needs well over 100 GB of disk): comment out the three `docker run ... quantization` lines at the end of the script before running it.

To verify SGX works end to end, run the Gramine hello world **inside** the SGX image (note the directory is `CI-Examples`, plural):

```sh
docker run --rm --privileged -it sgx-ipex-llm:2.2.0 bash -c "cd gramine/CI-Examples/helloworld && make SGX=1 && gramine-sgx helloworld"
```

In case you encounter errors related to Gramine, please refer to [its documentation](https://gramine.readthedocs.io/en/stable/) for debugging instructions.

### TDX Setup

> [!NOTE]
> **On-prem only.** This whole section builds and boots a TD guest on your own TDX host. On a CSP machine (e.g., the Azure `DC16es_v6` Confidential VM), the VM is already a TDX guest: skip to [Running TDX experiments](#running-tdx-experiments).

#### Prepare a TDX VM image
Use TDX guest tools to generate a TDX VM image. By default, we create a 300GB image but it should be at least 200GB (required for 70B Llama2 model). For more in depth treatment such as BIOS configuration for TDX, follow the instructions within the [Ubuntu's TDX](https://github.com/canonical/tdx) repository. In short, run:
```sh
cd tdx/guest-tools/image/
sudo ./create-td-image.sh
```
Update the `td_guest.xml` to point to the newly created image. Then, define and start the TD:
```sh
sudo virsh define td_guest.xml
sudo virsh start tdx
```
The default PW of user `ubuntu` is `123456`. The default port on which the VM will be available is 10022.
If you run into permission issues, it might be useful to copy the qcow2 file to libvirt's images:
```sh
sudo cp ~/confidential-llms-in-tees/tdx/guest-tools/image/tdx-guest-ubuntu-24.04-generic.qcow2 /var/lib/libvirt/images/
```
Consider creating an ssh key and copying it to the running TD for faster login.

#### Copy the repository to the VM
Initialize the repository in the VM exactly as outlined above in host setup or use `rsync` to copy the files to the VM:
```sh
rsync -avzog --exclude tdx/ -e 'ssh -p 10022' confidential-llms-in-tees/ tdx@localhost:~/confidential-llms-in-tees
```
SSH to the VM and run the host setup script:
```sh
ssh -p 10022 tdx@localhost
cd confidential-llms-in-tees
./host_setup.sh   # reads HUGGINGFACE_TOKEN from .env, or pass it inline
```
See [Hugging Face access token](#hugging-face-access-token) above for what permissions this token needs. Relogin to apply changes in groups. Finally, compile the docker container:
```sh
cd confidential-llms-in-tees/intel-extension-for-pytorch/
DOCKER_BUILDKIT=1 docker build -f examples/cpu/inference/python/llm/Dockerfile -t ipex-llm:2.3.100 .
```

#### Enable hugepages
In case you would like to measure the VMs with enabled 1GB hugepages, first modify Grub configuration in `/etc/default/grub` (e.g., for `<num_hugepages>=300`)
```sh
GRUB_CMDLINE_LINUX="nomodeset kvm_intel.tdx=1 default_hugepagesz=1G hugepagesz=1G hugepages=<num_hugepages> transparent_hugepages=always"
```
Then system.ctl `/etc/sysctl.conf`
```sh
vm.nr_hugepages=<num_hugepages>
```
Update grub
```sh
sudo update-grub
sudo reboot
```

To verify that the hugepages are indeed enabled, after reboot run:
```sh
cat /proc/meminfo | grep HugePages
```
which should report `<num_hugepages>`. 

Once rebooted, remember to use the hugepages version of the `.xml` VM definition file and modify it with `<num_hugepages>`. Then define and start this new VM:
```sh
sudo virsh define td_guest-hugepages.xml
sudo virsh tdx-hugepages
```
As of writing this, TDX does not support hugepages, so if you allocate 300GB of 1GB pages, it will still try to use 2MB pages and you might run out of memory. We used these pages only for VM measurements, and for TDX we used the default pages.

### Running baseline experiments

```sh
nohup ./run.sh baseline &
```

This will generate a folder under `results/` with the current date and time and add an entry into the experiment log. All generated files will have the form `baseline-system-in_size-out_size-vCPUs-numa-batch_size-model-data_type.txt`.

### Running TDX experiments
SSH to the TDX machine. On Azure that is the `DC16es_v6` VM (`ssh azureuser@<tdx-vm-ip>`); on-prem it is the TD guest created in [TDX Setup](#tdx-setup):
```sh
ssh -p 10022 tdx@localhost   # on-prem TD guest only
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
Run the docker image and then before running a workload activate environment:

```sh
source ./llm/tools/env_activate.sh
```

Run a workload - preferably on a single socket:
```sh
numactl -N 0,1 -m 0,1 -C 0-31 gramine-sgx LLM ~/llm/single_instance/run_generation.py --dtype bfloat16 -m meta-llama/Llama-2-7b-hf --input-tokens 512 --max-new-tokens 128 --num-iter 30 --num-warmup 5 --batch-size 1 --greedy --benchmark
```

### Quantizing models
To quantize the models, follow `genQuantLLamaModels.sh`.

### Processing Results

`processing/run_parser.py` gathers all iteration and token latencies from each
experiment and places them into a csv file. Pass it the **parent** `results`
directory (its glob only matches `.txt` files one level below the argument, so
passing a single `results/<date>-<time>` folder matches nothing). Run it from
`CPU/`:

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

## GPU
GPUs require vLLM. Follow (their installation instructions)[https://github.com/vllm-project/vllm] to enable them on your system.
You can then run the benchmark using:
```
./benchmark_vllm.sh
```
You can parse the produced logs using `parse.py` and plot using `plot_GPUs.py` (modify inside the names of your CSVs).

## RAG
Make sure you have your submodules initialized. Then, enter the RAG directory and apply the patch:
```
cd RAG/beir
git apply ../beir.patch
```
Start the elasticsearch database:
```
cd RAG
docker compose up elasticsearch
```
To build and run the benchmarks container:
```
docker compose run --rm --build rag
```
Within just run:
```
./run.sh
```
