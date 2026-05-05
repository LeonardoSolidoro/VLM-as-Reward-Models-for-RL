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
- Image frames (256x256 `.jpg`) from the `topview` camera.
- Ground-truth rewards.
- Success metrics.
- Actions taken.

### Directory Structure
```
data/metaworld/
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

### Collect Dataset
To collect the trajectory data for the tasks, run:
```bash
python src/collect_metaworld_data.py
```
You can modify the `rollouts_per_setting` variable in the script to adjust the number of trajectories collected.

### VLM Reward Inference
1. Copy the env template and edit it:
    ```bash
    cp .env.example .env
    ```
2. Set the local MLX model ID, for example:
    ```bash
    MODEL_NAME=mlx-community/Qwen3-VL-8B-Instruct-4bit
    MAX_TOKENS=80
    TEMPERATURE=0.0
    ```
3. Run the local test:
    ```bash
    python src/reward_function.py
    ```

The default MLX model is `mlx-community/Qwen3-VL-8B-Instruct-4bit`, which is a good balance for a 32GB Apple Silicon Mac. If you want faster reward sweeps, use `mlx-community/Qwen3-VL-4B-Instruct-4bit`; if memory pressure becomes an issue, use `mlx-community/Qwen3-VL-2B-Instruct-4bit`.
