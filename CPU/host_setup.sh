#!/bin/bash

# stop on failure
set -euo pipefail

# Load .env from the repo root if present; HUGGINGFACE_TOKEN can also be
# passed directly in the environment. Loaded before enabling trace (-x) so the
# token never ends up echoed into logs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    source "$SCRIPT_DIR/../.env"
fi
: "${HUGGINGFACE_TOKEN:?Set HUGGINGFACE_TOKEN in the environment or in .env (cp config.env .env)}"

set -x

# Create venv and login
# fresh images ship with empty apt lists, update before the first install
sudo apt update
sudo apt install -y python3-venv numactl lshw
python3 -m venv .venv
source .venv/bin/activate
pip install -U "huggingface_hub[cli]"
set +x
hf auth login --token "$HUGGINGFACE_TOKEN"
set -x

###### DOCKER #######
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER

set +x
echo ""
echo "=========================================================================="
echo "Docker is installed and $USER was added to the docker group. usermod alone"
echo "doesn't apply to this shell, so dropping you into a new shell with the"
echo "docker group active -- exit it to return to the shell that ran this script."
echo "=========================================================================="
exec newgrp docker