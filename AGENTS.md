# OpenCode Instructions

## Overview
Automates reward computation for Metaworld tasks using Visual Language Models (VLMs). The core workload is multi-view, frame-subsampled reward estimation.

## Key Configuration
- **`configs/configs.yaml`**: Source of truth for task descriptions, experiment views (`experiment_views`), context framing (`frames_in_context`), and prompt templates.

## Execution
- **Reward Collection**: Run `python src/collect_rewards.py`.
  - Input: `data/metaworld/`.
  - Output: `output/metaworld_rewards/<experiment_name>/<task>_<level>_rewards.json`.
  - Script uses `aiohttp` for parallel async VLM queries.
- **Metrics**: Use `src/compute_metrics.py` to analyze results.

## Critical Structure & Gotchas
- **Data Layout**: `data/metaworld/<task>/<level>/<rollout>/<view>_frame_<index>.jpg`.
- **Experiment Views**: `experiment_views` in `configs.yaml` MUST match the naming convention in the data directory (e.g., 'topview', 'corner'). If a view is missing, collection may fail or produce incomplete data.
- **Prompting**: The pipeline expects multi-view inputs mapped to `[IMG]` placeholders. When modifying prompts, ensure the number of images passed to `get_reward_score` matches the placeholders in the template.
