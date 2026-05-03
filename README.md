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
python scripts/collect_metaworld_data.py
```
You can modify the `rollouts_per_setting` variable in the script to adjust the number of trajectories collected.
