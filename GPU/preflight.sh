#!/bin/bash
#
# Pre-flight check for the GPU sweep (see GOOGLE_CLOUD.md).
#
# H100 spot time costs roughly $10/hour per VM, so every condition that could
# make a run fail late is checked here first, on the ground, at zero cost.
# Nothing in this script creates or modifies a cloud resource.
#
# Exits 0 only when the environment is trustworthy enough to provision.

set -uo pipefail

# Load .env from the repo root before enabling any trace, so HUGGINGFACE_TOKEN
# never ends up echoed into a log (same convention as CPU/host_setup.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    source "$SCRIPT_DIR/../.env"
fi

# Zones where Confidential VM with an H100 is supported. Order is the fallback
# order used when GCP_ZONE is unset.
SUPPORTED_ZONES=(us-central1-a us-east5-a europe-west4-c)

GPU_MACHINE_TYPE="${GPU_MACHINE_TYPE:-a3-highgpu-1g}"
GPU_IMAGE_PROJECT="${GPU_IMAGE_PROJECT:-ubuntu-os-cloud}"
GPU_IMAGE_FAMILY="${GPU_IMAGE_FAMILY:-ubuntu-2204-lts}"
MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"

FAILURES=0
USABLE_ZONE=""

pass() { printf '  \033[32mok\033[0m    %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; [ $# -gt 1 ] && printf '        -> %s\n' "$2"; FAILURES=$((FAILURES + 1)); }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$1"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$1"; }

########################################################################
head_ "1. Required binaries"
########################################################################

for bin in gcloud gh git python3 jq ssh curl; do
    if command -v "$bin" >/dev/null 2>&1; then
        pass "$bin ($(command -v "$bin"))"
    else
        case "$bin" in
            gcloud) hint="https://cloud.google.com/sdk/docs/install" ;;
            gh)     hint="https://cli.github.com/" ;;
            jq)     hint="sudo apt install jq" ;;
            *)      hint="sudo apt install $bin" ;;
        esac
        fail "$bin not found" "$hint"
    fi
done

# Everything below needs gcloud; bail out early rather than cascading errors.
if ! command -v gcloud >/dev/null 2>&1; then
    printf '\n\033[31mgcloud is missing -- cannot check any Google Cloud prerequisite.\033[0m\n'
    exit 1
fi

########################################################################
head_ "2. Google Cloud access"
########################################################################

ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null)"
if [ -n "$ACCOUNT" ]; then
    pass "authenticated as $ACCOUNT"
else
    fail "no active gcloud account" "run: gcloud auth login"
fi

if [ -z "${GCP_PROJECT:-}" ]; then
    GCP_PROJECT="$(gcloud config get-value project 2>/dev/null)"
    [ "$GCP_PROJECT" = "(unset)" ] && GCP_PROJECT=""
fi

if [ -z "$GCP_PROJECT" ]; then
    fail "no project selected" "set GCP_PROJECT in .env, or: gcloud config set project <PROJECT_ID>"
elif gcloud projects describe "$GCP_PROJECT" >/dev/null 2>&1; then
    pass "project $GCP_PROJECT is visible to this account"
else
    fail "project '$GCP_PROJECT' not found or not accessible" "check the id, and that $ACCOUNT has access to it"
fi

if [ -n "$GCP_PROJECT" ] && [ -n "$ACCOUNT" ]; then
    if gcloud services list --enabled --project "$GCP_PROJECT" --format='value(config.name)' 2>/dev/null \
        | grep -qx 'compute.googleapis.com'; then
        pass "Compute Engine API enabled"
    else
        fail "Compute Engine API not enabled" "gcloud services enable compute.googleapis.com --project $GCP_PROJECT"
    fi

    # testIamPermissions answers "may I create a VM?" without creating one.
    # Called over REST rather than via `gcloud projects test-iam-permissions`,
    # which does not exist in every gcloud release.
    NEEDED=(
        compute.instances.create
        compute.instances.delete
        compute.instances.get
        compute.instances.list
        compute.instances.setMetadata
        compute.disks.create
        compute.zoneOperations.get
    )
    PERM_JSON="$(printf '%s\n' "${NEEDED[@]}" | jq -R . | jq -sc '{permissions: .}')"
    GRANTED="$(curl -s -X POST \
        -H "Authorization: Bearer $(gcloud auth print-access-token 2>/dev/null)" \
        -H "Content-Type: application/json" \
        -d "$PERM_JSON" \
        "https://cloudresourcemanager.googleapis.com/v1/projects/$GCP_PROJECT:testIamPermissions" \
        2>/dev/null | jq -r '.permissions[]?' 2>/dev/null)"
    for perm in "${NEEDED[@]}"; do
        if echo "$GRANTED" | grep -qx "$perm"; then
            pass "$perm"
        else
            fail "$perm not granted" "ask a project admin for roles/compute.instanceAdmin.v1"
        fi
    done
fi

########################################################################
head_ "3. H100 quota (Confidential VM with GPU is Spot-only)"
########################################################################

# a3-highgpu-1g exists only under the Spot and flex-start provisioning models,
# so PREEMPTIBLE_NVIDIA_H100_GPUS -- not NVIDIA_H100_GPUS -- is the quota that
# actually gates this experiment.
if [ -n "$GCP_PROJECT" ] && [ -n "$ACCOUNT" ]; then
    # H100 quotas are not exposed by `gcloud compute regions describe` -- newer
    # GPU metrics live only in the Cloud Quotas API, where an unset quota comes
    # back as null rather than 0. Reading the legacy view reports FAIL even
    # after a grant, so query Cloud Quotas directly.
    QUOTA_ID="PREEMPTIBLE-NVIDIA-H100-GPUS-per-project-region"
    QUOTA_JSON="$(curl -s \
        -H "Authorization: Bearer $(gcloud auth print-access-token 2>/dev/null)" \
        "https://cloudquotas.googleapis.com/v1/projects/$GCP_PROJECT/locations/global/services/compute.googleapis.com/quotaInfos/$QUOTA_ID" \
        2>/dev/null)"

    printf '  %-18s %30s %8s\n' REGION METRIC LIMIT
    for zone in "${SUPPORTED_ZONES[@]}"; do
        region="${zone%-*}"
        limit="$(echo "$QUOTA_JSON" | jq -r --arg r "$region" '
            [.dimensionsInfos[]?
             | select((.applicableLocations // []) | index($r))
             | .details.value] | first // 0' 2>/dev/null)"
        [ "$limit" = "null" ] && limit=0
        limit="${limit:-0}"
        printf '  %-18s %30s %8s\n' "$region" PREEMPTIBLE_NVIDIA_H100_GPUS "$limit"
        if [ -z "$USABLE_ZONE" ] && awk "BEGIN{exit !($limit >= 1)}"; then
            USABLE_ZONE="$zone"
        fi
    done

    if [ -n "$USABLE_ZONE" ]; then
        pass "spot H100 quota available -- will use zone $USABLE_ZONE"
    else
        fail "PREEMPTIBLE_NVIDIA_H100_GPUS is 0 in all three supported regions" \
             "request an increase (see GOOGLE_CLOUD.md, 'Requesting a quota increase'); nothing else can run until it is granted"
    fi

    GLOBAL_LIMIT="$(gcloud compute project-info describe --project "$GCP_PROJECT" --format=json 2>/dev/null \
        | jq -r '.quotas[] | select(.metric=="GPUS_ALL_REGIONS") | (.limit - .usage)' 2>/dev/null)"
    if [ -n "$GLOBAL_LIMIT" ] && awk "BEGIN{exit !($GLOBAL_LIMIT >= 1)}"; then
        pass "GPUS_ALL_REGIONS has $GLOBAL_LIMIT GPU(s) of headroom"
    else
        fail "global GPUS_ALL_REGIONS quota exhausted" "request an increase alongside the regional quota"
    fi
fi

# An explicit GCP_ZONE overrides the auto-pick, but must be supported.
if [ -n "${GCP_ZONE:-}" ]; then
    if printf '%s\n' "${SUPPORTED_ZONES[@]}" | grep -qx "$GCP_ZONE"; then
        pass "GCP_ZONE=$GCP_ZONE is a supported Confidential-GPU zone"
        USABLE_ZONE="$GCP_ZONE"
    else
        fail "GCP_ZONE=$GCP_ZONE does not support Confidential VM with GPU" \
             "supported zones: ${SUPPORTED_ZONES[*]}"
    fi
fi

########################################################################
head_ "4. Machine type and image"
########################################################################

CHECK_ZONE="${USABLE_ZONE:-${SUPPORTED_ZONES[0]}}"
if [ -n "$GCP_PROJECT" ] && [ -n "$ACCOUNT" ]; then
    if gcloud compute machine-types describe "$GPU_MACHINE_TYPE" \
        --zone "$CHECK_ZONE" --project "$GCP_PROJECT" >/dev/null 2>&1; then
        pass "$GPU_MACHINE_TYPE is offered in $CHECK_ZONE"
    else
        fail "$GPU_MACHINE_TYPE not offered in $CHECK_ZONE" "try another zone from: ${SUPPORTED_ZONES[*]}"
    fi

    IMAGE_JSON="$(gcloud compute images describe-from-family "$GPU_IMAGE_FAMILY" \
        --project "$GPU_IMAGE_PROJECT" --format=json 2>/dev/null)"
    if [ -n "$IMAGE_JSON" ]; then
        IMAGE_NAME="$(echo "$IMAGE_JSON" | jq -r .name)"
        pass "image family $GPU_IMAGE_PROJECT/$GPU_IMAGE_FAMILY resolves to $IMAGE_NAME"
        if echo "$IMAGE_JSON" | jq -e '.guestOsFeatures[]?.type | select(. == "TDX_CAPABLE")' >/dev/null 2>&1; then
            pass "image is TDX_CAPABLE"
        else
            fail "image $IMAGE_NAME is not TDX_CAPABLE" "the confidential arm cannot boot from it"
        fi
    else
        fail "image family $GPU_IMAGE_PROJECT/$GPU_IMAGE_FAMILY did not resolve"
    fi
fi

########################################################################
head_ "5. Hugging Face access to the gated model"
########################################################################

# A token that authenticates but has not accepted Meta's licence returns 403.
# Catching that here is the single highest-value check in this script: the
# alternative is discovering it on a running $10/hour GPU.
if [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
    fail "HUGGINGFACE_TOKEN not set" "cp config.env .env and fill it in (see README, 'Hugging Face access token')"
else
    HF_CODE="$(curl -s -o /dev/null -w '%{http_code}' \
        -H "Authorization: Bearer $HUGGINGFACE_TOKEN" \
        "https://huggingface.co/api/models/$MODEL" 2>/dev/null)"
    case "$HF_CODE" in
        200) pass "token is valid and has access to $MODEL" ;;
        401) fail "Hugging Face rejected the token (401)" "regenerate it at https://huggingface.co/settings/tokens" ;;
        403) fail "token is valid but lacks access to $MODEL (403)" \
                  "accept the licence at https://huggingface.co/$MODEL while logged in as the token owner" ;;
        000) fail "could not reach huggingface.co" "check outbound network access" ;;
        *)   fail "unexpected HTTP $HF_CODE from huggingface.co for $MODEL" ;;
    esac
fi

########################################################################
head_ "6. GitHub access (for committing results and opening the PR)"
########################################################################

if command -v gh >/dev/null 2>&1; then
    if gh auth status >/dev/null 2>&1; then
        pass "gh is authenticated as $(gh api user --jq .login 2>/dev/null)"
    else
        fail "gh is not authenticated" "run: gh auth login"
    fi
fi

########################################################################
printf '\n'
if [ "$FAILURES" -eq 0 ]; then
    printf '\033[32mPre-flight passed.\033[0m Provision with GCP_ZONE=%s -- see GOOGLE_CLOUD.md.\n' "$CHECK_ZONE"
    printf 'Run the smoke test before the full sweep:\n'
    printf '    ./benchmark_vllm.sh cgpu --smoke\n'
    exit 0
else
    printf '\033[31mPre-flight failed with %d problem(s).\033[0m Fix them before provisioning --\n' "$FAILURES"
    printf 'each one would otherwise surface on a running H100 at roughly $10/hour.\n'
    exit 1
fi
