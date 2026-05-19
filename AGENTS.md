# OpenCode Instructions

## Overview
This repository automates reward computation for Metaworld tasks using Visual Language Models (VLMs).

## Key Configuration
- **`configs/configs.yaml`**: The single source of truth for task descriptions, camera views, and sampling logic (step sizes, interval).

## Execution
- **Reward Collection**: Run `python src/collect_rewards.py`.
  - This reads from `data/metaworld` and writes to `output/metaworld_rewards`.
  - The script uses `aiohttp` to run multiple VLM queries in parallel (via `asyncio`).
- **Metrics**: Use `src/compute_metrics.py` to analyze the collected rewards.

## Structure & Gotchas
- **Data Directory**: Expects `data/metaworld/<task>/<level>/<rollout>/*.jpg`.
- **Camera Views**: The `camera_names` list in `configs.yaml` is critical. If a requested camera view is missing from a rollout, the collection will fail.
- **Dependencies**: `requirements.txt` includes `aiohttp` and `opencv-python`. Ensure environment has these.
