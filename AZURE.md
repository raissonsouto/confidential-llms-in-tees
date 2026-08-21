# Deploying the Azure VMs for benchmarks

This doc covers provisioning the Azure VMs for the comparisons,
the actual benchmark setup and execution is covered in the main
[README.md](README.md).

## Table of contents

- [Deploying the Azure VMs for benchmarks](#deploying-the-azure-vms-for-benchmarks)
  - [Table of contents](#table-of-contents)
  - [Virtual machines used](#virtual-machines-used)
  - [Prerequisites](#prerequisites)
  - [Configuration](#configuration)
  - [Resource group](#resource-group)
  - [Checking quota before you provision](#checking-quota-before-you-provision)
  - [SGX VM (native and SGX arms)](#sgx-vm-native-and-sgx-arms)
  - [TDX VM](#tdx-vm)
  - [Pausing between runs](#pausing-between-runs)
  - [Cleaning up](#cleaning-up)
  - [Troubleshooting](#troubleshooting)
    - [Requesting a quota increase](#requesting-a-quota-increase)

## Virtual machines used

| | **SGX VM** | **TDX VM** |
|---|---|---|
| Size | `Standard_DC16s_v3` | `Standard_DC16es_v6` |
| Family | DCsv3 | DCesv6 |
| vCPUs | 16 | 16 |
| Memory | 128 GiB | 64 GiB |
| OS disk | 128 GB | 128 GB |
| TEE | Intel SGX* | Intel TDX |
| Image | Ubuntu 24.04 LTS (Gen2 `server` SKU) | Ubuntu 24.04 LTS (`cvm` SKU) |
| Arms it runs | Native baseline **and** SGX | TDX |

> \* The native baseline shares the SGX VM because SGX is opt-in per process:
> `gramine-sgx` enters the enclave only for the workload it launches, so the
> same machine runs the same docker image with and without the TEE. TDX
> instead encrypts the entire VM and cannot be disabled from inside, so the
> TDX machine cannot host a native run.

The memory gap doesn't affect batch-1 (Llama-2-7B in bf16, ~14 GB, fits both
VMs and the SGX EPC), but at batch 64 it's the binding constraint: 512-token
inputs OOM-kill on the TDX VM's 64 GiB, and 2048-token inputs OOM-kill even
on the 128 GiB machine. The comparable quantity is each machine's
TEE-vs-native ratio, not absolute latencies, since the VMs are different
hardware generations. All runs use bfloat16 and the same `run.sh`.

## Prerequisites

- **Pay-As-You-Go subscription.** Free trial/free-account subscriptions don't
  have enough quota for any of these VM families and aren't eligible for quota
  increases at all. Convert to PAYG first.
- **Azure CLI installed and logged in**: `az login`, then confirm you're on the
  right subscription with `az account show --output table`. If you have more
  than one subscription under the same login, double check with
  `az account list --output table` and `az account set --subscription <id>`.
  It's easy to be logged into an old/unrelated subscription that doesn't have
  write access to anything.
- Region availability is restricted for the confidential-computing families:
  Microsoft has stated it won't deploy new DCsv2/DCsv3/DCdsv3 VMs into new
  regions, so check the [products-by-region page](https://azure.microsoft.com/global-infrastructure/services/?products=virtual-machines)
  before picking a region, for both the DCsv3 and the v6 families.

## Configuration

The commands below read their names/region from `.env` at the repo root.
Create it once from the `config.env` template and adjust to taste:

```sh
cp config.env .env
# edit .env, then load it into the current shell:
source .env
```

`.env` is gitignored because it also holds `HUGGINGFACE_TOKEN`.

## Resource group

Creates the container everything else (VMs, disks, NICs) gets deployed into:

```sh
az group create --name $AZURE_RESOURCE_GROUP --location $AZURE_REGION
```

The group's location is only where its metadata lives. A group can hold
resources from any region, and each VM deploys into its own region
(`SGX_VM_REGION` and `TDX_VM_REGION` in `.env`), since quota for the two VM
families is often granted in different regions.

## Checking quota before you provision

Quota is checked per VM-family, per region, and is commonly **0 by default**
even on a paid subscription for specialty families (SGX/TDX/confidential
compute), so don't assume quota exists just because the subscription is PAYG.
Check each VM's family in that VM's region:

```sh
# SGX VM family (DCSv3), in the SGX VM's region
az vm list-usage --location $SGX_VM_REGION --query "[?contains(localName,'DCSv3') || contains(localName,'Total Regional')]" --output table
# TDX VM family (DCEV6), in the TDX VM's region
az vm list-usage --location $TDX_VM_REGION --query "[?contains(localName,'DCEV6') || contains(localName,'Total Regional')]" --output table
```

Also check the SKU itself isn't blocked for your subscription independently of
quota, since brand-new subscriptions can hit `NotAvailableForSubscription` even
before quota comes into play:

```sh
az vm list-skus --location $SGX_VM_REGION --size DC16s_v3 --resource-type virtualMachines --all --output table
az vm list-skus --location $TDX_VM_REGION --size DC16es_v6 --resource-type virtualMachines --all --output table
```

If quota is 0 or the SKU is restricted, see
[Requesting a quota increase](#requesting-a-quota-increase) in the
Troubleshooting section.

Quota increases don't bill anything on their own, so there's no need to
"return" them.

## SGX VM (native and SGX arms)

This section creates the SGX VM. It doesn't need a
`--security-type`/confidential-VM flag, because SGX is
exposed as a CPU feature on a normal VM. It does require a Generation 2
image, and the Ubuntu 24.04 `server` SKU is Gen2 (note: Microsoft's SGX docs
officially list Ubuntu 20.04/22.04 Gen2. 24.04 boots as Gen2 all the same,
but if the SGX driver stack misbehaves, 22.04
`0001-com-ubuntu-server-jammy:22_04-lts-gen2` is the documented fallback).

Both VMs are created with a 128 GB OS disk (`--os-disk-size-gb`): the image
default is 30 GB, which fills up during the first benchmark run (the docker
image plus the Llama-2-7B weights alone exceed it) and fails with
`No space left on device`.

```sh
az vm create \
  --resource-group $AZURE_RESOURCE_GROUP \
  --location $SGX_VM_REGION \
  --name $SGX_VM_NAME \
  --image Canonical:ubuntu-24_04-lts:server:latest \
  --size $SGX_VM_SIZE \
  --os-disk-size-gb 128 \
  --admin-username $AZURE_ADMIN_USER \
  --generate-ssh-keys \
  --public-ip-sku Standard
```

## TDX VM

This section creates the TDX VM. It needs `--security-type ConfidentialVM`
plus OS disk encryption and boot-security flags, and the CVM image SKU.
**Confirm the current flags
against Microsoft's own quickstart before running this**, since
confidential-VM CLI options have changed across Azure CLI versions:
[Create an Azure confidential VM in the Azure portal](https://learn.microsoft.com/en-us/azure/confidential-computing/quick-create-confidential-vm-portal),
[DCesv6-series docs](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/general-purpose/dcesv6-series).

```sh
az vm create \
  --resource-group $AZURE_RESOURCE_GROUP \
  --location $TDX_VM_REGION \
  --name $TDX_VM_NAME \
  --image Canonical:ubuntu-24_04-lts:cvm:latest \
  --size $TDX_VM_SIZE \
  --os-disk-size-gb 128 \
  --admin-username $AZURE_ADMIN_USER \
  --generate-ssh-keys \
  --security-type ConfidentialVM \
  --os-disk-security-encryption-type DiskWithVMGuestState \
  --enable-secure-boot true \
  --enable-vtpm true \
  --public-ip-sku Standard
```

## Pausing between runs

A stopped VM still bills unless it's **deallocated**, so always use
`az vm deallocate`, and confirm the power state says `VM deallocated`:

```sh
az vm deallocate --resource-group $AZURE_RESOURCE_GROUP --name $SGX_VM_NAME
az vm deallocate --resource-group $AZURE_RESOURCE_GROUP --name $TDX_VM_NAME
az vm list --resource-group $AZURE_RESOURCE_GROUP --show-details --query "[].{name:name, power:powerState}" -o table
```

## Cleaning up

To delete a single VM but keep the rest, remove it together with its
attached resources:

```sh
az vm delete --resource-group $AZURE_RESOURCE_GROUP --name $SGX_VM_NAME --yes
az vm delete --resource-group $AZURE_RESOURCE_GROUP --name $TDX_VM_NAME --yes
```

Then list what's left over and delete orphaned disks/NICs/IPs by name:

```sh
az resource list --resource-group $AZURE_RESOURCE_GROUP -o table
```

When the experiments are done, delete the whole resource group instead. This
removes the VMs and everything created alongside them (disks, NICs, public
IPs, NSGs), which `az vm delete` alone would leave behind:

```sh
# make sure results are copied off the VMs first, e.g.:
scp -r $AZURE_ADMIN_USER@<vm-ip>:~/confidential-llms-in-tees/CPU/results ./results-backup/

az group delete --name $AZURE_RESOURCE_GROUP --yes --no-wait
```

`--no-wait` returns immediately; check progress with
`az group show --name $AZURE_RESOURCE_GROUP` (a `ResourceGroupNotFound` error
means the deletion finished).

## Troubleshooting

### Requesting a quota increase

If [checking quota](#checking-quota-before-you-provision) shows 0 or the SKU
is restricted, request an increase. The generic `az quota` extension often
returns nothing for these families (it did for us, even after installing it).
The reliable path is a support ticket:

```sh
az extension add --name support

# Find the exact problem-classification ID once:
az support services list --query "[?contains(displayName,'quota')]" -o table
az support services problem-classifications list \
  --service-name "06bfd9d3-516b-d5c6-5802-169c800dec89" \
  --query "[?contains(DisplayName,'Compute-VM')]" -o table

az support in-subscription tickets create \
  --ticket-name "dcsv3-quota-increase" \
  --title "Quota increase: Standard DCSv3 Family vCPUs to 16 in East US" \
  --description "Requesting an increase of the Standard DCSv3 Family vCPUs quota from 0 to 16 in East US to run CPU LLM inference benchmarks." \
  --problem-classification "/providers/Microsoft.Support/services/06bfd9d3-516b-d5c6-5802-169c800dec89/problemClassifications/e12e3d1d-7fa0-af33-c6d0-3c50df9658a3" \
  --severity "minimal" \
  --advanced-diagnostic-consent "Yes" \
  --contact-first-name "<first>" --contact-last-name "<last>" \
  --contact-email "<you@example.com>" --contact-country "<ISO3166-alpha3>" \
  --contact-language "en-us" --contact-timezone "<Windows timezone name>" \
  --contact-method "email" \
  --quota-change-version "1.0" \
  --quota-change-requests "[{region:'$SGX_VM_REGION',payload:'{VMFamily:standardDCSv3Family,NewLimit:16}'}]"
```

Repeat with `standardDCEV6Family` and `$TDX_VM_REGION` for the TDX VM (that is the family name
deployment errors report for `Standard_DC16es_v6`, not "DCESv6"). Approval
isn't guaranteed or instant: for a brand-new personal subscription, budget
same-day to a couple of business days rather than expecting it immediately.
The `QuotaExceeded` deployment error also carries a direct portal link to a
pre-filled quota request for the exact family and region, which can be
faster than the CLI ticket.
