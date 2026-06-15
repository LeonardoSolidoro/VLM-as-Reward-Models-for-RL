#!/bin/bash
set -e

echo "=================================================="
echo "1. Installing System Monitoring & Git"
echo "=================================================="
# Update apt and install htop, git, tmux, and nvtop (excellent for GPU VRAM monitoring)
apt-get update -y
apt-get install -y htop git tmux
apt-get install -y nvtop || echo "nvtop not available in this apt repo, skipping..."

echo "=================================================="
echo "2. Creating Persistent Conda Environment in /workspace"
echo "=================================================="
# RUNPOD PRO-TIP: We must create the environment directly inside /workspace/envs
# because any environment created in the default /opt/conda folder will be 
# permanently deleted when the Pod restarts!
cd /workspace
mkdir -p /workspace/envs

# Ensure conda is in path if it was installed previously but the terminal forgot it
if [ -d "/workspace/miniconda3" ] && ! command -v conda &> /dev/null; then
    export PATH="/workspace/miniconda3/bin:$PATH"
fi

# Install Miniconda into /workspace if it doesn't exist
if ! command -v conda &> /dev/null; then
    echo "Conda not found! Installing Miniconda to /workspace/miniconda3..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p /workspace/miniconda3
    rm miniconda.sh
    export PATH="/workspace/miniconda3/bin:$PATH"
    echo "Miniconda installed successfully."
fi

# Automatically accept Anaconda Terms of Service to prevent non-interactive script crashes
if command -v conda &> /dev/null; then
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
fi

ENV_PREFIX="/workspace/envs/VLM_RM_flashattn"

if [ ! -d "$ENV_PREFIX" ]; then
    echo "Creating Conda environment at $ENV_PREFIX"
    conda create -y --prefix $ENV_PREFIX python=3.11
else
    echo "Environment already exists at $ENV_PREFIX. Skipping creation."
fi

# Initialize conda for the bash script session
eval "$(conda shell.bash hook)"
conda activate $ENV_PREFIX

echo "=================================================="
echo "3. Installing PyTorch & Project Requirements"
echo "=================================================="
# We assume you have cloned your repo into /workspace/VLM-as-Reward-Models-for-RL
# If not, the script will wait here for you to clone it or manually upload requirements.txt
REPO_DIR="/workspace/VLM-as-Reward-Models-for-RL"
if [ -d "$REPO_DIR" ]; then
    cd $REPO_DIR
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        echo "requirements.txt not found in $REPO_DIR"
    fi
else
    echo "Directory $REPO_DIR not found."
    echo "Please clone your repository first: git clone <your-repo-url> $REPO_DIR"
    echo "Then manually run: pip install -r requirements.txt"
fi

echo "=================================================="
echo "4. Installing Pre-compiled Flash Attention"
echo "=================================================="
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

echo "=================================================="
echo "RunPod Setup Complete!"
echo "To activate your persistent environment, run:"
echo "conda activate /workspace/envs/VLM_RM_flashattn"
echo "=================================================="
