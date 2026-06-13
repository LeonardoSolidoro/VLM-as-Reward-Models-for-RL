#!/bin/bash
set -e

ENV_NAME="VLM_RM_flashattn"

echo "=================================================="
echo "1. Creating Conda Environment: $ENV_NAME"
echo "=================================================="
conda create -y -n $ENV_NAME python=3.11

# Initialize conda for the bash script session
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "=================================================="
echo "2. Installing Core Requirements (PyTorch 2.5.1)"
echo "=================================================="
pip install -r requirements.txt

echo "=================================================="
echo "3. Installing Pre-compiled Flash Attention"
echo "=================================================="
# We install the wheel directly to completely bypass local C++ compilation, 
# preventing OS crashes and avoiding cross-device link errors
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

echo "=================================================="
echo "Setup Complete!"
echo "You can now activate the environment using:"
echo "conda activate $ENV_NAME"
echo "=================================================="
