#!/bin/bash
#
# Guest-side setup for a Google Cloud H100 benchmark VM. Run this ON the
# instance, not on your workstation. See GOOGLE_CLOUD.md for how to create the
# instance in the first place.
#
#     ./gcp_vm_setup.sh gpu     # baseline: stock driver, no CC mode
#     ./gcp_vm_setup.sh cgpu    # confidential: TDX guest + GPU CC mode
#
# The script reboots once (the NVIDIA driver and the LKCA initramfs change both
# need it). Re-run it after the reboot; it picks up where it left off.

set -euo pipefail

SYSTEM="${1:-}"
if [ "$SYSTEM" != "gpu" ] && [ "$SYSTEM" != "cgpu" ]; then
    echo "usage: $0 {gpu|cgpu}" >&2
    exit 1
fi

# Load .env before enabling trace so HUGGINGFACE_TOKEN is never echoed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    source "$SCRIPT_DIR/../.env"
fi
: "${HUGGINGFACE_TOKEN:?Set HUGGINGFACE_TOKEN in the environment or in .env (cp config.env .env)}"

MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
VLLM_VERSION="${VLLM_VERSION:-0.9.2}"
VLLM_REPO="${VLLM_REPO:-$HOME/vllm}"
STAMP_DIR="$HOME/.gpu_setup_stamps"
mkdir -p "$STAMP_DIR"

done_with()  { [ -f "$STAMP_DIR/$1" ]; }
mark_done()  { touch "$STAMP_DIR/$1"; }

########################################################################
# 1. Base packages and NVIDIA driver
########################################################################
# Driver 580 or later is required for GPU confidential computing. Installed on
# both arms so the only difference between them is the TEE, not the driver.

if ! done_with driver; then
    set -x
    sudo apt-get update --yes
    sudo apt-get install --yes \
        "linux-headers-$(uname -r)" build-essential libxml2 libncurses5-dev \
        pkg-config libvulkan1 gcc-12 python3-venv python3-pip git curl jq

    DISTRO=$(. /etc/os-release && echo "${ID}${VERSION_ID//./}")
    curl -fsSL -o /tmp/cuda-keyring.deb \
        "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt-get update --yes
    sudo apt-get install --yes nvidia-open-580 || sudo apt-get install --yes cuda-drivers-580
    set +x
    mark_done driver
    NEED_REBOOT=1
fi

########################################################################
# 2. Confidential-computing mode (cgpu only)
########################################################################
# Two guest-side changes are needed before the H100 will come up in CC mode:
#
#   * the Linux Kernel Crypto API modules must load before the nvidia module,
#     because the driver uses them for the encrypted SPDM session it sets up
#     with the GPU;
#   * persistence must be uvm-persistence-mode -- in CC mode, tearing the
#     driver context down and back up renegotiates that session, which is both
#     slow and a source of run-to-run variance in a benchmark.
#
# The baseline arm is deliberately left stock so it stays a clean control.

if [ "$SYSTEM" = "cgpu" ] && ! done_with ccmode; then
    set -x
    echo "install nvidia /sbin/modprobe ecdsa_generic; /sbin/modprobe ecdh; /sbin/modprobe --ignore-install nvidia" \
        | sudo tee /etc/modprobe.d/nvidia-lkca.conf
    sudo update-initramfs -u

    sudo test -f /usr/lib/systemd/system/nvidia-persistenced.service && \
        sudo sed -i "s/no-persistence-mode/uvm-persistence-mode/g" \
            /usr/lib/systemd/system/nvidia-persistenced.service
    sudo systemctl daemon-reload
    set +x
    mark_done ccmode
    NEED_REBOOT=1
fi

if [ "${NEED_REBOOT:-0}" = "1" ]; then
    echo ""
    echo "=========================================================================="
    echo "Rebooting to activate the driver and any CC-mode changes."
    echo "Reconnect and run this script again to finish setup:"
    echo "    ./gcp_vm_setup.sh $SYSTEM"
    echo "=========================================================================="
    sudo reboot
    exit 0
fi

########################################################################
# 3. Verify the TEE is actually on
########################################################################
# This is the check the whole experiment rests on. A cgpu VM that silently
# comes up with CC status OFF produces a second baseline run, and the measured
# "overhead" would be noise around zero.

echo ""
echo "=== GPU ==="
nvidia-smi

if [ "$SYSTEM" = "cgpu" ]; then
    echo ""
    echo "=== Confidential computing status ==="
    sudo nvidia-smi conf-compute -srs 1 || true
    CC_STATUS="$(sudo nvidia-smi conf-compute -f 2>&1 || true)"
    echo "$CC_STATUS"
    sudo nvidia-smi conf-compute -grs || true

    if ! echo "$CC_STATUS" | grep -qi 'CC status.*ON'; then
        echo ""
        echo "ERROR: the GPU is NOT in confidential computing mode." >&2
        echo "This VM would produce baseline numbers under a confidential label." >&2
        echo "Check that the instance was created with --confidential-compute-type=TDX" >&2
        echo "and that the driver is 580 or newer, then re-run this script." >&2
        exit 1
    fi

    echo ""
    echo "=== TDX guest ==="
    # A TDX guest exposes the TDX guest device and reports it in the kernel log.
    ls -l /dev/tdx_guest 2>/dev/null || echo "(no /dev/tdx_guest)"
    sudo dmesg | grep -i -m5 'tdx' || echo "(no TDX lines in dmesg)"
fi

########################################################################
# 4. vLLM and the model
########################################################################

if ! done_with vllm; then
    python3 -m venv "$HOME/.venv"
    # shellcheck disable=SC1091
    source "$HOME/.venv/bin/activate"
    pip install -U pip
    # vLLM 0.9.2 declares only transformers>=4.51.1, so a fresh install resolves
    # to a far newer major that breaks it: transformers >= 4.54 registers an
    # "aimv2" config itself, and vLLM's own registration then dies with
    # "'aimv2' is already used by a Transformers config". Pin a contemporaneous
    # release instead of taking whatever pip resolves to.
    TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.53.2}"
    pip install "vllm==${VLLM_VERSION}" "transformers==${TRANSFORMERS_VERSION}" "huggingface_hub[cli]"

    # The pip wheel does not ship benchmarks/, which is what benchmark_vllm.sh
    # invokes, so the matching tag is cloned separately.
    [ -d "$VLLM_REPO" ] || git clone --depth 1 -b "v${VLLM_VERSION}" \
        https://github.com/vllm-project/vllm.git "$VLLM_REPO"

    hf auth login --token "$HUGGINGFACE_TOKEN"
    # Pull the weights now rather than inside the first timed run.
    hf download "$MODEL"
    mark_done vllm
fi

########################################################################
# 5. Hardware snapshot
########################################################################
# Mirrors what CPU/run.sh captures (lscpu / lshw / numactl) so a GPU result
# folder documents its own machine the same way the CPU ones do.

SNAP="$SCRIPT_DIR/hwinfo-$SYSTEM"
mkdir -p "$SNAP"
nvidia-smi -q                        > "$SNAP/nvidia-smi.out"  2>&1 || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv \
                                     > "$SNAP/gpu.csv"         2>&1 || true
lscpu                                > "$SNAP/lscpu.out"       2>&1 || true
uname -a                             > "$SNAP/uname.out"       2>&1 || true
free -h                              > "$SNAP/free.out"        2>&1 || true
if [ "$SYSTEM" = "cgpu" ]; then
    sudo nvidia-smi conf-compute -f  > "$SNAP/conf-compute.out" 2>&1 || true
fi
# shellcheck disable=SC1091
source "$HOME/.venv/bin/activate"
python3 -c "import vllm, torch; print('vllm', vllm.__version__); print('torch', torch.__version__)" \
                                     > "$SNAP/versions.out"    2>&1 || true

echo ""
echo "=========================================================================="
echo "Setup complete for '$SYSTEM'. Hardware snapshot in $SNAP"
echo ""
echo "Smoke test (smallest config -- batch 1, input 128):"
echo "    source ~/.venv/bin/activate && ./benchmark_vllm.sh $SYSTEM --smoke"
echo ""
echo "Full sweep:"
echo "    source ~/.venv/bin/activate && nohup ./benchmark_vllm.sh $SYSTEM > sweep-$SYSTEM.log 2>&1 &"
echo "=========================================================================="
