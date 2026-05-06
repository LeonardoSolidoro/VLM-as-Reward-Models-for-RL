# VLM as Reward Models for RL

This repository focuses on evaluating Vision-Language Models (VLMs) as reward signals for Reinforcement Learning (RL) tasks, specifically using the **Meta-World** benchmark.

## Experimental Setup

The main benchmark consists of 50 simulated robotic manipulation tasks. We evaluate VLM-based rewards across various task categories:
- **Basic manipulation**: Reaching and pushing.
- **Object interaction**: Drawer opening, door opening, and button pressing.
- **Complex manipulation**: Peg insertion and pick-and-place.

## Dataset Collection

We collect rollouts with three levels of task completion:
- **Expert**: Generated using Meta-World's scripted policies.
- **Near-Expert**: Expert actions with injected Gaussian noise.
- **Random**: Random action sampling.

For each rollout, we store:
- Image frames.
- Ground-truth rewards.
- Success metrics.
- Actions taken.

### Directory Structure
```
data/metaworld/{camera_name}
└── {task}/
    └── {level}/
        └── rollout_{i}/
            ├── frame_000.jpg
            ├── frame_001.jpg
            └── ...
            └── metadata.json
```

## Setup

### Installation
1. Create and activate the Conda environment:
   ```bash
   conda create -n VLM_RM python=3.11 -y
   conda activate VLM_RM
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Collect Dataset
To collect trajectory data for the tasks from Meta-World, run:
```bash
python src/collect_metaworld_data.py
```
This script generates expert, near-expert, and random rollouts with frames and metadata.

### 2. Set Up VLM Server
The reward calculation depends on a running VLM server (e.g., using `vllm`). Start the server with:
```bash
vllm serve cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048 \
    --enforce-eager
```

### 3. Compute VLM Rewards
To evaluate the VLM on the collected datasets and generate potential reward scores, run:
```bash
python src/collect_rewards.py
```

### 4. Evaluate Metrics
To compute correlation metrics (Pearson correlation and preference alignment) between VLM rewards and ground-truth rewards, run:
```bash
python src/compute_metrics.py
```
