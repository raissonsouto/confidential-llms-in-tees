# Deploying the Google Cloud VMs for GPU benchmarks

This doc covers provisioning the Google Cloud VMs for the GPU comparison,
the actual benchmark setup and execution is covered in the main
[README.md](README.md). The CPU arms run on Azure and are covered in
[AZURE.md](AZURE.md).

## Table of contents

- [Deploying the Google Cloud VMs for GPU benchmarks](#deploying-the-google-cloud-vms-for-gpu-benchmarks)
  - [Table of contents](#table-of-contents)
  - [Virtual machines used](#virtual-machines-used)
  - [Prerequisites](#prerequisites)
  - [Configuration](#configuration)
  - [Pre-flight check](#pre-flight-check)
    - [Automated](#automated)
    - [Manual](#manual)
  - [Checking quota before you provision](#checking-quota-before-you-provision)
  - [Baseline GPU VM](#baseline-gpu-vm)
  - [Confidential GPU VM](#confidential-gpu-vm)
  - [Enabling and verifying GPU confidential computing](#enabling-and-verifying-gpu-confidential-computing)
  - [Surviving Spot preemption](#surviving-spot-preemption)
  - [Pausing between runs](#pausing-between-runs)
  - [Cleaning up](#cleaning-up)
  - [Troubleshooting](#troubleshooting)
    - [Requesting a quota increase](#requesting-a-quota-increase)
    - [ZONE_RESOURCE_POOL_EXHAUSTED](#zone_resource_pool_exhausted)
    - [CC status reports OFF](#cc-status-reports-off)

## Virtual machines used

| | **Baseline GPU VM** | **Confidential GPU VM** |
|---|---|---|
| Machine type | `a3-highgpu-1g` | `a3-highgpu-1g` |
| GPU | 1× NVIDIA H100 80 GB | 1× NVIDIA H100 80 GB |
| vCPUs | 26 | 26 |
| Memory | 234 GB | 234 GB |
| Boot disk | 200 GB | 200 GB |
| TEE | none | Intel TDX + GPU CC mode |
| Provisioning | Spot\* | Spot\* |
| Image | Ubuntu 22.04 LTS (`ubuntu-2204-lts`) | Ubuntu 22.04 LTS (`ubuntu-2204-lts`) |
| Arms it runs | `gpu` | `cgpu` |

> \* Not a choice. `a3-highgpu-1g` is offered **only** under the Spot and
> flex-start provisioning models — there is no on-demand SKU — and Confidential
> VM with Intel TDX additionally
> [does not support reservations](https://docs.cloud.google.com/confidential-computing/confidential-vm/docs/create-a-confidential-vm-instance-with-gpu).
> Both arms are therefore Spot, which at least makes them symmetric, but either
> VM can be preempted mid-sweep. See
> [Surviving Spot preemption](#surviving-spot-preemption).

Unlike the Azure setup, where one VM hosts both the native and SGX arms, **two
separate instances are required here**. GPU confidential-computing mode is
fixed when the instance is created (`--confidential-compute-type=TDX`) and
cannot be toggled from inside the guest, so a confidential VM cannot produce a
baseline number and vice versa.

Everything else about the two instances is deliberately identical — same
machine type, same zone, same image, same driver version, same vLLM build — so
the only variable between the arms is the TEE.

## Prerequisites

- **A project with H100 quota.** Confidential VM with GPU needs
  `PREEMPTIBLE_NVIDIA_H100_GPUS` in the region plus global `GPUS_ALL_REGIONS`
  headroom. Both default to **0**, including on billing-enabled projects, and a
  grant is not instant. See
  [Requesting a quota increase](#requesting-a-quota-increase).
- **gcloud CLI installed and logged in**: `gcloud auth login`, then select the
  project with `gcloud config set project <PROJECT_ID>` and confirm with
  `gcloud config list`. It is easy to be authenticated but pointed at no
  project at all, in which case every command below fails with a confusing
  permission error rather than a clear one.
- **Compute Engine API enabled**:
  `gcloud services enable compute.googleapis.com`.
- **A supported zone.** Confidential VM with an H100 exists in exactly three
  zones — `us-central1-a`, `us-east5-a`, `europe-west4-c`
  ([supported configurations](https://docs.cloud.google.com/confidential-computing/confidential-vm/docs/supported-configurations)).
  Anywhere else, instance creation fails regardless of quota.
- **A supported image.** Only `ubuntu-2204-lts` and `cos-tdx-113-lts` are
  supported with GPUs. Google is explicit that other images tagged
  `TDX_CAPABLE` are *not* supported in that combination — note this is Ubuntu
  22.04, whereas the CPU arms in [AZURE.md](AZURE.md) run 24.04.
- **Budget.** `a3-highgpu-1g` spot is roughly **$10/hour**. The full sweep is
  about 2-4 hours per arm including driver installation and model download, so
  budget on the order of $40-80 for both arms.

## Configuration

The commands below read their names/zone from `.env` at the repo root.
Create it once from the `config.env` template and adjust to taste:

```sh
cp config.env .env
# edit .env, then load it into the current shell:
source .env
```

`.env` is gitignored because it also holds `HUGGINGFACE_TOKEN`.

## Pre-flight check

H100 spot time bills by the second, so every condition that could make a run
fail late is worth checking first, on the ground, at zero cost. Do not
provision until the environment checks out.

There are two ways to do this: run the script, or work through the same checks
by hand.

### Automated

`preflight.sh` covers all of it — binaries, gcloud authentication, project
visibility, Compute Engine API, the specific IAM permissions needed to create
and delete an instance, quota in all three supported regions, machine-type and
image availability, `TDX_CAPABLE` on the image, Hugging Face access to the
gated Llama-2 repo, and GitHub authentication:

```sh
cd GPU
./preflight.sh
```

It creates nothing and exits non-zero with an actionable message on the first
problem.

### Manual

The same checks, one at a time, if you would rather see each result yourself or
are debugging a specific failure.

**Authentication and project**

```sh
gcloud auth list                       # an ACTIVE account must be listed
gcloud projects describe $GCP_PROJECT  # must resolve
gcloud services list --enabled --project $GCP_PROJECT | grep compute.googleapis.com
```

**Permissions** — answers "may I create a VM?" without creating one:

```sh
curl -s -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d '{"permissions":["compute.instances.create","compute.instances.delete","compute.disks.create"]}' \
  "https://cloudresourcemanager.googleapis.com/v1/projects/$GCP_PROJECT:testIamPermissions"
```

Every permission you asked about must come back in the response. Anything
missing is not granted; ask a project admin for `roles/compute.instanceAdmin.v1`.

**Quota** — see [Checking quota before you provision](#checking-quota-before-you-provision) below.

**Machine type and image**

```sh
gcloud compute machine-types describe a3-highgpu-1g --zone=$GCP_ZONE
gcloud compute images describe-from-family ubuntu-2204-lts \
  --project=ubuntu-os-cloud --format="value(name,guestOsFeatures)"
```

The image's `guestOsFeatures` must include `TDX_CAPABLE`, or the confidential
arm cannot boot from it.

**Hugging Face access** — a token that authenticates fine but whose owner has
not accepted Meta's licence returns **403** on the model download, and the
natural place to discover that is twenty minutes into setting up a running
$10/hour GPU:

```sh
curl -s -o /dev/null -w '%{http_code}\n' \
  -H "Authorization: Bearer $HUGGINGFACE_TOKEN" \
  https://huggingface.co/api/models/meta-llama/Llama-2-7b-hf
```

`200` is what you want. `401` means the token is bad; `403` means the token is
valid but the licence has not been accepted — see
[Hugging Face access token](README.md#hugging-face-access-token).

## Checking quota before you provision

Quota is per-metric and per-region, and is **0 by default** for H100 even on a
paid project, so don't assume it exists. Because `a3-highgpu-1g` is Spot-only,
the metric that actually gates this experiment is
`PREEMPTIBLE_NVIDIA_H100_GPUS`, **not** `NVIDIA_H100_GPUS`:

> [!IMPORTANT]
> Do **not** check this with `gcloud compute regions describe`. That command
> lists the legacy quota metrics only, and H100 is not among them — it returns
> nothing at all for `PREEMPTIBLE_NVIDIA_H100_GPUS`, which is indistinguishable
> from a limit of 0 if you are not expecting it. Newer GPU metrics live in the
> Cloud Quotas API, where an unset quota comes back as `null`.

```sh
curl -s -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://cloudquotas.googleapis.com/v1/projects/$GCP_PROJECT/locations/global/services/compute.googleapis.com/quotaInfos/PREEMPTIBLE-NVIDIA-H100-GPUS-per-project-region" \
  | jq -r '.dimensionsInfos[] | "\(.applicableLocations) -> \(.details.value)"'
```

Regions you have never been granted quota in are grouped into one entry with
`value: null`; a granted region appears on its own with its limit. You need at
least 1 in one of `us-central1`, `us-east5` or `europe-west4`.

There is also a global cap on GPUs of all types across all regions, which is
easy to overlook because it is on the project, not the region. This one *is*
still in the legacy view:

```sh
gcloud compute project-info describe --format=json \
  | jq -r '.quotas[] | select(.metric=="GPUS_ALL_REGIONS")'
```

If `limit` is 0 for either, see
[Requesting a quota increase](#requesting-a-quota-increase). Quota increases
don't bill anything on their own — only running instances do.

## Baseline GPU VM

This creates the non-confidential arm. It is the same machine type as the
confidential VM, just without `--confidential-compute-type`:

```sh
gcloud compute instances create $GPU_VM_NAME \
  --project=$GCP_PROJECT \
  --zone=$GCP_ZONE \
  --machine-type=$GPU_MACHINE_TYPE \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --image-project=$GPU_IMAGE_PROJECT \
  --image-family=$GPU_IMAGE_FAMILY \
  --boot-disk-size=$GPU_BOOT_DISK_SIZE \
  --boot-disk-type=pd-balanced
```

`--maintenance-policy=TERMINATE` is mandatory: GPU instances cannot live
migrate. `--boot-disk-size` is 200 GB rather than the 30 GB in Google's example
command — the NVIDIA driver, CUDA runtime, torch and the ~13.5 GB of
Llama-2-7B weights do not fit in 30 GB, and the run dies with
`No space left on device` partway through.

## Confidential GPU VM

Same shape, plus `--confidential-compute-type=TDX`:

```sh
gcloud compute instances create $CGPU_VM_NAME \
  --project=$GCP_PROJECT \
  --zone=$GCP_ZONE \
  --machine-type=$GPU_MACHINE_TYPE \
  --confidential-compute-type=TDX \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --image-project=$GPU_IMAGE_PROJECT \
  --image-family=$GPU_IMAGE_FAMILY \
  --boot-disk-size=$GPU_BOOT_DISK_SIZE \
  --boot-disk-type=pd-balanced
```

**Confirm the current flags against Google's own quickstart before running
this**, since Confidential VM CLI options have shifted between gcloud releases:
[Create a Confidential VM instance with GPU](https://docs.cloud.google.com/confidential-computing/confidential-vm/docs/create-a-confidential-vm-instance-with-gpu),
[Supported configurations](https://docs.cloud.google.com/confidential-computing/confidential-vm/docs/supported-configurations).

## Enabling and verifying GPU confidential computing

Creating the instance with `--confidential-compute-type=TDX` gives you a TDX
*guest*. Getting the *GPU* into confidential-computing mode needs three more
things inside the guest, all handled by `gcp_vm_setup.sh`:

1. **Driver 580 or newer.** Earlier drivers have no CC mode.
2. **The Linux Kernel Crypto API loaded before the `nvidia` module.** The
   driver uses `ecdsa_generic`/`ecdh` for the encrypted session it negotiates
   with the GPU, so they must be present when it initialises:
   ```sh
   echo "install nvidia /sbin/modprobe ecdsa_generic; /sbin/modprobe ecdh; /sbin/modprobe --ignore-install nvidia" \
     | sudo tee /etc/modprobe.d/nvidia-lkca.conf
   sudo update-initramfs -u
   ```
3. **`uvm-persistence-mode`**, so the driver context — and with it the
   negotiated session — is not torn down and rebuilt between processes:
   ```sh
   sudo sed -i "s/no-persistence-mode/uvm-persistence-mode/g" \
     /usr/lib/systemd/system/nvidia-persistenced.service
   sudo systemctl daemon-reload
   ```

Then reboot and copy the repo across:

```sh
gcloud compute scp --recurse --zone=$GCP_ZONE \
  GPU .env $CGPU_VM_NAME:~/confidential-llms-in-tees/
gcloud compute ssh $CGPU_VM_NAME --zone=$GCP_ZONE
cd ~/confidential-llms-in-tees/GPU
./gcp_vm_setup.sh cgpu     # reboots once; re-run afterwards to finish
```

> [!IMPORTANT]
> Verify the TEE is actually on before spending any GPU time on measurements:
> ```sh
> sudo nvidia-smi conf-compute -f     # must print: CC status: ON
> sudo nvidia-smi conf-compute -grs   # Confidential Compute GPUs Ready state: ready
> ```
> A confidential VM whose GPU quietly came up with `CC status: OFF` produces a
> second baseline run wearing a confidential label, and the "overhead" you go
> on to report is noise around zero. `gcp_vm_setup.sh` refuses to continue in
> that case; if you set the VM up by hand, check it by hand. `-srs 1` sets the
> ready state, which has to be redone after each reboot.

Do **not** flash the GPU firmware on a Confidential VM instance — Google
explicitly warns this causes instability and crashes.

## Surviving Spot preemption

`a3-highgpu-1g` only exists as Spot, so preemption is a normal operating
condition, not an edge case. Two things make it survivable:

- **The sweep is resumable.** `benchmark_vllm.sh` skips any configuration whose
  `.json` already exists, so a preemption costs the configuration in flight and
  nothing else. Point `RESULTS_DIR` at the existing folder when you resume:
  ```sh
  RESULTS_DIR=results_cgpu_2026-08-21_14-03-11 ./benchmark_vllm.sh cgpu
  ```
- **`--instance-termination-action=STOP`** (above) stops the instance instead of
  deleting it, so the boot disk — driver, venv, downloaded weights — survives.
  Restart it with `gcloud compute instances start` and re-run the sweep;
  everything except the in-flight configuration is still there.

Run the sweep detached so an SSH drop doesn't kill it, the same way the CPU
arms do:

```sh
source ~/.venv/bin/activate
nohup ./benchmark_vllm.sh cgpu > sweep-cgpu.log 2>&1 &
```

Copy results off **as soon as each arm finishes**, not at the end of the day —
a preempted-and-stopped instance is cheap, but a deleted one takes its disk
with it.

## Pausing between runs

A stopped instance stops billing for CPU/GPU but **keeps billing for its boot
disk**, which at 200 GB is minor but not zero. Stop it while you are not using
it, and confirm the status:

```sh
gcloud compute instances stop $CGPU_VM_NAME --zone=$GCP_ZONE
gcloud compute instances list --zones=$GCP_ZONE \
  --format="table(name,status,machineType.basename())"
```

## Cleaning up

Copy the results off first — the boot disk goes away with the instance:

```sh
gcloud compute scp --recurse --zone=$GCP_ZONE \
  $CGPU_VM_NAME:~/confidential-llms-in-tees/GPU/results_cgpu_* ./results/cgpu/
gcloud compute scp --recurse --zone=$GCP_ZONE \
  $GPU_VM_NAME:~/confidential-llms-in-tees/GPU/hwinfo-gpu ./results/gpu/
```

Then delete both instances. `gcloud compute instances delete` removes the
boot disk along with the instance by default (unlike `az vm delete`), so there
is usually nothing orphaned to chase:

```sh
gcloud compute instances delete $GPU_VM_NAME  --zone=$GCP_ZONE --quiet
gcloud compute instances delete $CGPU_VM_NAME --zone=$GCP_ZONE --quiet
```

Confirm nothing is left running or holding a disk:

```sh
gcloud compute instances list
gcloud compute disks list
```

## Troubleshooting

### Requesting a quota increase

If [checking quota](#checking-quota-before-you-provision) shows 0, request an
increase. You need **both** the regional spot-H100 metric and the global GPU
metric — raising only one still blocks instance creation:

| Metric | Scope | Request |
|---|---|---|
| `PREEMPTIBLE_NVIDIA_H100_GPUS` | one of `us-central1`, `us-east5`, `europe-west4` | ≥ 1 |
| `GPUS_ALL_REGIONS` | global | ≥ 1 |

The CLI path:

```sh
# List the current values and their quota IDs
gcloud alpha services quota list \
  --service=compute.googleapis.com \
  --consumer=projects/$GCP_PROJECT \
  --filter="metric:compute.googleapis.com/preemptible_nvidia_h100_gpus"
```

In practice the console is the more reliable route for GPU families: open
**IAM & Admin → Quotas & System Limits**, filter for `H100`, tick the
`PREEMPTIBLE_NVIDIA_H100_GPUS` row for your region, and click **Edit Quotas**.
Repeat for `GPUS_ALL_REGIONS`.

Approval is neither guaranteed nor instant. H100 capacity is scarce and
requests from projects with no prior GPU usage are frequently declined or
trimmed; budget several business days, and expect to justify the workload in
the request description. If the request is declined, the practical fallbacks
are to ask through your organisation's Google Cloud account team, or to run the
GPU arm on another provider that offers confidential H100 instances.

### ZONE_RESOURCE_POOL_EXHAUSTED

Quota is permission to ask; it is not a reservation. Spot H100 capacity in the
three TDX-capable zones is genuinely scarce, and creation can fail even with
quota granted. Try the zones in turn:

```sh
for ZONE in us-central1-a us-east5-a europe-west4-c; do
  echo "== trying $ZONE"
  gcloud compute instances create $CGPU_VM_NAME --zone=$ZONE ... && break
done
```

Confidential VM with TDX cannot use reservations, so the alternative to
retrying is the flex-start provisioning model, which queues the request until
capacity appears instead of failing immediately:

```sh
gcloud beta compute instance-templates create cgpu-flex \
  --provisioning-model=FLEX_START \
  --confidential-compute-type=TDX \
  --machine-type=$GPU_MACHINE_TYPE \
  --maintenance-policy=TERMINATE \
  --image-project=$GPU_IMAGE_PROJECT \
  --image-family=$GPU_IMAGE_FAMILY \
  --reservation-affinity=none \
  --boot-disk-size=$GPU_BOOT_DISK_SIZE \
  --instance-termination-action=DELETE \
  --max-run-duration=6h \
  --project=$GCP_PROJECT
```

`--max-run-duration` must be between 600 and 604800 seconds, and the instance
is **deleted** when it expires — so copy results off before the deadline.

### CC status reports OFF

The instance is a TDX guest but the GPU is not in confidential mode. In order
of likelihood:

1. The driver is older than 580 — check `nvidia-smi --query-gpu=driver_version --format=csv`.
2. `/etc/modprobe.d/nvidia-lkca.conf` is missing, or `update-initramfs -u` was
   not run after writing it, or the VM has not been rebooted since.
3. The instance was created without `--confidential-compute-type=TDX`. Confirm
   with:
   ```sh
   gcloud compute instances describe $CGPU_VM_NAME --zone=$GCP_ZONE \
     --format="value(confidentialInstanceConfig)"
   ```
   This cannot be fixed in place — delete the instance and recreate it.
