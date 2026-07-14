# **View-Robust Vision-Language Rewards for Online Reinforcement Learning**

This project investigates whether vision-language models can `estimate robotic task progress` and `provide reward signals for online reinforcement learning` in ManiSkill3, particularly under `wrist-mounted camera motion`. It extends [*Vision Language Models Are In-Context Value Learners*](https://doi.org/10.48550/arXiv.2411.04549) by evaluating camera robustness, improving view robustness through cross-view contrastive fine-tuning, and using learned visual rewards for online policy training.

## **Project Outline**

The project has three stages. Qwen3-VL-8B-Instruct is the main trainable model; Stage 1 additionally benchmarks Qwen3-VL-32B-Instruct and Gemini 3.1 Flash-Lite.

### **Stage 1: Static Reproduction and Camera-Robustness Gap**

- Reproduce zero-shot and few-shot task-progress prediction under static camera motion (Qwen3-VL-8B-Instruct, Qwen3-VL-32B-Instruct, Gemini-3.1-Flash-Lite)
- Fine-tune Qwen3-VL-8B-Instruct on static camera data
- Quantify the performance drop caused by wrist-mounted camera motion on the fine-tuned Qwen3-VL-8B-Instruct model, motivating Stage 2 

### **Stage 2: View-Robust Task Progress Fine-Tuning**

- Implement `supervised fine-tuning` (on wrist-mounted camera data) and `cross-view contrastive fine-tuning` (on synchronized static + wrist-mounted camera data) for task progress prediction under wrist-mounted camera motion
- Evaluate both methods on the same held-out wrist-mounted camera data

### **Stage 3: Learned Visual Rewards for Online RL**

- Attach a lightweight MLP reward head to the contrastively adapted visual representation and refine it with policy-induced hard negatives
- Train a PickCube policy online using the learned visual reward

## **Reported Results**

| Experiment | Result |
|---|---:|
| Supervised task-progress MAE (wrist-mounted camera) | 5.66 percentage points |
| Contrastive task-progress MAE (wrist-mounted camera) | 3.70 percentage points |
| Peak online-RL training evaluation (20 episodes) | 80% success |
| Initial actor held-out evaluation (200 episodes) | 60.5% success |
| Best actor held-out evaluation (200 episodes) | 77.0% success |

Cross-view contrastive fine-tuning reduced wrist-mounted task-progress MAE from `5.66` to `3.70` percentage points, which is an absolute improvement of `1.96 points` and a relative reduction of `34.6%`.

For online RL, rewards were produced by a contrastively adapted visual encoder, attention pooler, and lightweight MLP head refined with policy-induced hard negatives. The run began from a bootstrap actor with `50% success` in its initial 20-episode training evaluation and reached a peak of `80%`. On 200 separate held-out episodes, the best visual-reward actor achieved `77.0% success`, compared with `60.5%` for the initial actor, which is a gain of `16.5 percentage points`.

The commands below reproduce the complete protocol. Exact values may vary with API outputs, random seeds, hardware, and dependency versions. Compact summaries of the completed runs are stored in [`results/`](results/).

## **Setup**

The main experiments were run with an `NVIDIA A100 SXM4 80 GB`.

Set up the environment: 

```bash
python3 -m venv VLM_RM
source VLM_RM/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Verify CUDA before training:

```bash
python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
print(torch.cuda.get_device_name(0))
PY
```

Required demonstrations (PickCube-v1, PushCube-v1 and PegInsertionSide-v1 are already contained in this repository):

```text
demos/<task>/motionplanning/trajectory.h5
demos/<task>/motionplanning/trajectory.json
```

Run all commands from the `project root`. 

Data generation uses CPU PhysX and SAPIEN rendering. The A100 is used for model training, reward inference, and RL updates.

## **Stage 1: Static Reproduction and Camera-Robustness Gap**

### **Method**

Stage 1 first reproduces task progress prediction under the paper's standard static camera setting. It then tests whether a model fine-tuned in that setting transfers to a wrist-mounted camera.

1. `Reproduce zero-shot and few-shot task-progress prediction.` Evaluate Qwen3-VL-8B-Instruct, Qwen3-VL-32B-Instruct, and Gemini-3.1-Flash-Lite on static camera expert demonstrations. Each rollout contains 20 temporal frames. The first frame is supplied as a 0% progress anchor, the remaining 19 frames are shuffled, and the VLM predicts one completion percentage per frame. One-shot and two-shot runs prepend labeled expert examples from the same static camera dataset.

2. `Fine-tune Qwen3-VL-8B on static-camera data.` Split 500 expert rollouts per task into 400 training, 50 validation, and 50 test trajectories. The vision encoder remains frozen while QLoRA adapters on the LLM-side attention projections learn to generate the 19 progress labels. 

3. `Measure the camera-robustness gap.` Render the same demonstrations and simulator states from a wrist-mounted camera (each time step is synchronized between static and wrist-mounted camera), then construct the same train, validation, and test split using the same seed. Evaluate the static-trained adapter on both static and wrist-mounted versions of the 150 held-out trajectories without any additional training. Since the task states, temporal labels, and split are unchanged, the difference in MAE and value-order correlation isolates the effect of `camera motion`. This performance drop motivates the `view-robust training` introduced in Stage 2.

For the API reproduction, we report `value-order correlation (VOC)`, which is the Spearman correlation between predicted progress and the true temporal frame order. The first 50 rollouts per task are used:

- **0-shot:** 150 evaluated rollouts.
- **1-shot:** 147 evaluated rollouts; one rollout per task is reserved as context.
- **2-shot:** 144 evaluated rollouts; two rollouts per task are reserved as context.

For the fine-tuned model, we report MAE, RMSE, accuracy within five percentage points, and VOC over the held-out split.

### **Results**

The static-camera zero-shot and few-shot reproduction produced:

| Pretrained model | 0-shot VOC | 1-shot VOC | 2-shot VOC |
|---|---:|---:|---:|
| Gemini-3.1-Flash-Lite | **0.802** | **0.730** | 0.691 |
| Qwen3-VL-8B-Instruct | 0.247 | 0.364 | 0.448 |
| Qwen3-VL-32B-Instruct | 0.700 | 0.684 | **0.718** |

After supervised fine-tuning on static-camera data, Qwen3-VL-8B achieved high in-domain VOC but degraded when the same adapter was evaluated on matched wrist-mounted views:

| Training camera | Evaluation camera | Mean VOC ↑ | Median VOC ↑ |
|---|---|---:|---:|
| Static | Static | **0.991** | **0.995** |
| Static | Wrist-mounted | 0.820 | 0.889 |

The mean VOC drop of `0.171` demonstrates that strong static-camera fine-tuning does not by itself provide camera robustness.

**Result file:** [`results/gvl_reproduction.json`](results/gvl_reproduction.json)

### **Code Map**

| Step | Code |
|---|---|
| Replay demonstrations under each camera | [`generate_progress_data.py`](src/vlm_reward/data/generate_progress_data.py) |
| Build API prompts and in-context examples | [`prompts.py`](src/vlm_reward/baselines/prompts.py) |
| Run and score API experiments | [`runner.py`](src/vlm_reward/baselines/runner.py), [`api.py`](src/vlm_reward/baselines/api.py), [`metrics.py`](src/vlm_reward/baselines/metrics.py) |
| Build deterministic fine-tuning splits | [`build_progress_dataset.py`](src/vlm_reward/data/build_progress_dataset.py) |
| Train the static-camera QLoRA adapter | [`train_progress.py`](src/vlm_reward/training/train_progress.py) |
| Evaluate the adapter across cameras | [`infer_progress.py`](src/vlm_reward/evaluation/infer_progress.py), [`progress_metrics.py`](src/vlm_reward/evaluation/progress_metrics.py) |
| Reproduction settings | [`paper_reproduction.yaml`](configs/baselines/paper_reproduction.yaml) |

### **Run**

#### **1. Generate Static-Camera Data**

```bash
vlm-generate-progress-data \
  --data-root data \
  --demo-root demos \
  --camera-modes static \
  --tasks PickCube-v1 PushCube-v1 PegInsertionSide-v1 \
  --view topview \
  --num-rollouts 500 \
  --num-frames 20 \
  --seed 42 \
  --deterministic
```

#### **2. Run the Static Zero-Shot and Few-Shot Reproduction**

Configure the OpenAI-compatible API once:

```bash
cp .env.example .env
```

Set `VLM_API_URL` and `VLM_API_KEY` in `.env`. The command below explicitly supplies each endpoint model identifier:

```bash
for ENTRY in \
  "Qwen3-VL-8B-Instruct|Qwen/Qwen3-VL-8B-Instruct" \
  "Qwen3-VL-32B-Instruct|Qwen/Qwen3-VL-32B-Instruct" \
  "Gemini-3.1-Flash-Lite|google/gemini-3.1-flash-lite"; do
  MODEL_LABEL="${ENTRY%%|*}"
  MODEL_ID="${ENTRY#*|}"

  for SHOTS in 0 1 2; do
    NAME="${MODEL_LABEL}-static-${SHOTS}-Shot"

    MODEL_NAME="$MODEL_ID" vlm-baseline-run \
      --config configs/baselines/paper_reproduction.yaml \
      --data-root data/static \
      --in-context-examples "$SHOTS" \
      --experiment-name "$NAME"

    vlm-baseline-metrics \
      --results-dir "output/$NAME" \
      --output "output/$NAME/metrics.json"
  done
done
```

Each run writes raw predictions, `metrics.json`, `voc_histogram.png`, and `voc_by_task.png` under `output/<experiment-name>/`.

#### **3. Build the Static Fine-Tuning Dataset**

```bash
vlm-build-progress-data \
  --data-root data \
  --output-root finetune_data/static \
  --camera-mode static \
  --level expert \
  --tasks PickCube-v1 PushCube-v1 PegInsertionSide-v1 \
  --view topview \
  --num-frames 20 \
  --train-per-task 400 \
  --val-per-task 50 \
  --test-per-task 50 \
  --tiny-per-task 2 \
  --seed 42
```

#### **4. Fine-Tune Qwen3-VL-8B on Static Data**

```bash
vlm-train-progress \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --train-jsonl finetune_data/static/train.jsonl \
  --val-jsonl finetune_data/static/val.jsonl \
  --config configs/configs.yaml \
  --output-dir finetuning_output/Qwen3-VL-8B-Progress-Supervised-Static/lora_weights \
  --num-train-epochs 3 \
  --learning-rate 2e-4 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --dataloader-num-workers 4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --seed 42 \
  --bf16 \
  --qlora
```

Evaluate the adapter on the held-out static split:

```bash
vlm-infer-progress \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --adapter-dir finetuning_output/Qwen3-VL-8B-Progress-Supervised-Static/lora_weights \
  --jsonl finetune_data/static/test.jsonl \
  --config configs/configs.yaml \
  --output-json finetuning_output/Qwen3-VL-8B-Progress-Supervised-Static/predictions_static.json \
  --max-new-tokens 512 \
  --batch-size 4 \
  --dataloader-num-workers 4 \
  --bf16

vlm-score-progress \
  --input-json finetuning_output/Qwen3-VL-8B-Progress-Supervised-Static/predictions_static.json \
  --output-dir finetuning_output/Qwen3-VL-8B-Progress-Supervised-Static/results_static
```

#### **5. Evaluate the Static-Trained Model under Wrist-Mounted Camera Motion**

Render the same demonstrations from the wrist-mounted camera and build the matched split:

```bash
vlm-generate-progress-data \
  --data-root data \
  --demo-root demos \
  --camera-modes moving_mounted \
  --tasks PickCube-v1 PushCube-v1 PegInsertionSide-v1 \
  --view topview \
  --num-rollouts 500 \
  --num-frames 20 \
  --seed 42 \
  --deterministic

vlm-build-progress-data \
  --data-root data \
  --output-root finetune_data/moving_mounted \
  --camera-mode moving_mounted \
  --level expert \
  --tasks PickCube-v1 PushCube-v1 PegInsertionSide-v1 \
  --view topview \
  --num-frames 20 \
  --train-per-task 400 \
  --val-per-task 50 \
  --test-per-task 50 \
  --tiny-per-task 2 \
  --seed 42
```

Evaluate the same static-trained adapter without further fine-tuning:

```bash
vlm-infer-progress \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --adapter-dir finetuning_output/Qwen3-VL-8B-Progress-Supervised-Static/lora_weights \
  --jsonl finetune_data/moving_mounted/test.jsonl \
  --config configs/configs.yaml \
  --output-json finetuning_output/Qwen3-VL-8B-Progress-Supervised-Static/predictions_moving_mounted.json \
  --max-new-tokens 512 \
  --batch-size 4 \
  --dataloader-num-workers 4 \
  --bf16

vlm-score-progress \
  --input-json finetuning_output/Qwen3-VL-8B-Progress-Supervised-Static/predictions_moving_mounted.json \
  --output-dir finetuning_output/Qwen3-VL-8B-Progress-Supervised-Static/results_moving_mounted
```

Compare `accuracy_metrics.json` and `voc_metrics.json` between `results_static/` and `results_moving_mounted/`. This is the Stage 1 camera-robustness gap.

## **Stage 2: View-Robust Task-Progress Fine-Tuning**

Stage 2 compares `supervised and cross-view contrastive fine-tuning` for task-progress prediction under wrist-mounted camera motion. Both methods use Qwen3-VL-8B-Instruct and are evaluated on the same held-out wrist-mounted-camera split.

### **Method**

Every trajectory has one initial image and 19 shuffled query images. Task progress is defined to be between 0% and 100%.

The supervised baseline freezes the Qwen vision encoder and trains `LLM-side LoRA parameters` to generate progress text.

The contrastive model pairs matching frames from static and wrist-mounted cameras. It trains `all-linear LoRA parameters`, including vision layers, with:

```text
total loss = text cross-entropy + contrastive weight × symmetric InfoNCE
```

An `attention pooler (Multi-Head Attention)` converts each image's visual tokens from all batches into one embedding (this is required by contrastive fine-tuning). Matching cross-camera frames are positives. Nearby frames from the same trajectory are masked as false negatives. Both models are evaluated on the same held-out wrist-mounted-camera split.

### **Results**

Both models were evaluated on the same 2,850 held-out wrist-mounted frames:

| Method | MAE ↓ | RMSE ↓ | Within 5 points ↑ | Mean VOC ↑ |
|---|---:|---:|---:|---:|
| Supervised fine-tuning | 5.66 | 11.17 | 70.18% | 0.928 |
| Cross-view contrastive fine-tuning | **3.70** | **7.59** | **78.63%** | **0.967** |

Cross-view contrastive fine-tuning reduced MAE by `34.6%` relative to supervised fine-tuning while also improving value-order correlation.

**Result file:** [`results/finetuning_comparison_wrist_mounted.json`](results/finetuning_comparison_wrist_mounted.json)

### **Code Map**

| Step | Code |
|---|---|
| Build single-camera progress JSONL | [`build_progress_dataset.py`](src/vlm_reward/data/build_progress_dataset.py) |
| Build paired-camera JSONL | [`build_contrastive_dataset.py`](src/vlm_reward/data/build_contrastive_dataset.py) |
| Supervised dataset and prompt formatting | [`progress_dataset.py`](src/vlm_reward/training/progress_dataset.py), [`progress_format.py`](src/vlm_reward/training/progress_format.py) |
| Contrastive paired dataset | [`contrastive_dataset.py`](src/vlm_reward/training/contrastive_dataset.py) |
| Supervised LoRA | [`train_progress.py`](src/vlm_reward/training/train_progress.py) |
| Cross-view LoRA and InfoNCE | [`train_contrastive.py`](src/vlm_reward/training/train_contrastive.py), [`contrastive.py`](src/vlm_reward/models/contrastive.py) |
| Inference and metrics | [`infer_progress.py`](src/vlm_reward/evaluation/infer_progress.py), [`progress_metrics.py`](src/vlm_reward/evaluation/progress_metrics.py) |

### **Run**

Stage 1 already generated both camera views and built both single-camera datasets. When following the complete protocol, skip the next two blocks and continue at **Build static/wrist-mounted pairs**. If starting directly from Stage 2, generate the images and datasets with:

```bash
vlm-generate-progress-data \
  --data-root data \
  --demo-root demos \
  --camera-modes static moving_mounted \
  --tasks PickCube-v1 PushCube-v1 PegInsertionSide-v1 \
  --view topview \
  --num-rollouts 500 \
  --num-frames 20 \
  --seed 42 \
  --deterministic
```

Then build the two single-camera datasets:

```bash
for camera in static moving_mounted; do
  vlm-build-progress-data \
    --data-root data \
    --output-root "finetune_data/$camera" \
    --camera-mode "$camera" \
    --level expert \
    --tasks PickCube-v1 PushCube-v1 PegInsertionSide-v1 \
    --view topview \
    --num-frames 20 \
    --train-per-task 400 \
    --val-per-task 50 \
    --test-per-task 50 \
    --tiny-per-task 2 \
    --seed 42
done
```

Build static/wrist-mounted pairs:

```bash
vlm-build-contrastive-data \
  --data-root data \
  --output-root finetune_data_contrastive \
  --secondary-camera-mode moving_mounted \
  --level expert \
  --tasks PickCube-v1 PushCube-v1 PegInsertionSide-v1 \
  --view topview \
  --num-frames 20 \
  --train-per-task 400 \
  --val-per-task 50 \
  --test-per-task 50 \
  --tiny-per-task 2 \
  --train-secondary-fraction 0.70 \
  --seed 42
```

Train the supervised wrist-mounted camera baseline:

```bash
vlm-train-progress \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --train-jsonl finetune_data/moving_mounted/train.jsonl \
  --val-jsonl finetune_data/moving_mounted/val.jsonl \
  --config configs/configs.yaml \
  --output-dir finetuning_output/Qwen3-VL-8B-Progress-Supervised-Moving-Mounted/lora_weights \
  --num-train-epochs 3 \
  --learning-rate 2e-4 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --dataloader-num-workers 4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --seed 42 \
  --bf16 \
  --qlora
```

Train the paired-camera contrastive model:

```bash
vlm-train-contrastive \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --train-jsonl finetune_data_contrastive/train.jsonl \
  --val-jsonl finetune_data_contrastive/val.jsonl \
  --config configs/configs.yaml \
  --output-dir finetuning_output/Qwen3-VL-8B-Progress-Contrastive/lora_weights \
  --num-train-epochs 3 \
  --learning-rate 1e-4 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --save-strategy epoch \
  --eval-strategy epoch \
  --dataloader-num-workers 4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --contrastive-weight 1.0 \
  --seed 42 \
  --bf16 \
  --qlora
```

Evaluate both on the same wrist-mounted test set:

```bash
TEST_JSONL=finetune_data/moving_mounted/test.jsonl

vlm-infer-progress \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --adapter-dir finetuning_output/Qwen3-VL-8B-Progress-Supervised-Moving-Mounted/lora_weights \
  --jsonl "$TEST_JSONL" \
  --config configs/configs.yaml \
  --output-json finetuning_output/Qwen3-VL-8B-Progress-Supervised-Moving-Mounted/predictions.json \
  --max-new-tokens 512 \
  --batch-size 4 \
  --dataloader-num-workers 4 \
  --bf16

vlm-score-progress \
  --input-json finetuning_output/Qwen3-VL-8B-Progress-Supervised-Moving-Mounted/predictions.json \
  --output-dir finetuning_output/Qwen3-VL-8B-Progress-Supervised-Moving-Mounted/results

vlm-infer-progress \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --adapter-dir finetuning_output/Qwen3-VL-8B-Progress-Contrastive/lora_weights \
  --jsonl "$TEST_JSONL" \
  --config configs/configs.yaml \
  --output-json finetuning_output/Qwen3-VL-8B-Progress-Contrastive/predictions.json \
  --max-new-tokens 512 \
  --batch-size 4 \
  --dataloader-num-workers 4 \
  --bf16

vlm-score-progress \
  --input-json finetuning_output/Qwen3-VL-8B-Progress-Contrastive/predictions.json \
  --output-dir finetuning_output/Qwen3-VL-8B-Progress-Contrastive/results
```

## **Stage 3: Learned Visual Rewards for Online RL**

Stage 3 `converts the contrastively adapted visual representation into a fast reward model`, `refines it against policy-induced errors`, and uses it to `train a PickCube policy online`.

### **3.1 Visual Reward Model**

#### **Method**

We first used autoregressive task progress predictions directly as rewards ($r_t = p_t - p_{t-1}$), but the signal was very noisy and did not train a reliable policy. With the increase in task progress predictions as rewards, the policy was only able to converge to a maximum of 10% success rate and plateued there. The code of this can be found under `src/historical_progress_reward/`. 

Therefore, we changed direction and trained a `separate visual reward model` on ManiSkill's normalized dense reward.

1. `Build paired offline reward data.` Generate expert, partial, random, and regressing trajectories for PickCube, PushCube, and PegInsertionSide. Every sampled simulator state is rendered from both a static camera and a wrist-mounted camera, with the same ManiSkill normalized dense reward in `[0, 1]` assigned to both views. Expert trajectories cover successful behavior, partial trajectories stop demonstrations early, random trajectories cover low-quality behavior, and regressing trajectories move forward before returning to earlier states.

2. `Contrastively adapt Qwen to reward estimation.` Train a separate Qwen3-VL-8B LoRA adapter on the paired reward data. This produces a `camera-robust visual representation for instantaneous reward estimation`; it is separate from the task-progress adapter trained in Stage 2!

3. `Train a fast frozen-encoder reward head.` Freeze the reward-adapted vision encoder and attention pooler, run them once over the PickCube dataset, and cache one embedding per image. Train a small MLP on these embeddings to predict a scalar reward in `[0, 1]`. A weighted sampler balances five reward ranges. We select the checkpoint that performs well across all reward ranges, thereby preventing the many low-reward frames from dominating the evaluation. This head `replaces slow autoregressive decoding during RL`.

4. `Collect policy-induced errors.` Train two 60,000-step probe policies using the frozen reward head. We then run the trained policies again and record the images they produce, and compare the model's predicted reward with ManiSkill3's actual reward. We save the so-called `hard negative transitions`, that is when the model predicts at least `0.25` but the environment reward is at most `0.10`. These are the states most likely to cause `reward hacking` (policy finds a bad transition that leads to a high reward predicted by the VLM).

5. `Jointly refine the visual reward model.` Mix the original offline PickCube data with the probe-policy hard negative transitions and train jointly the vision LoRA parameters, attention pooler, and MLP head. Sampling is balanced by reward bin, online samples receive `1.5x` weight, and hard negatives receive `2x` weight. We select the checkpoint that performs well across all reward ranges, especially on policy-generated states, while avoiding high rewards for unsuccessful states. We also reject models that predict nearly the same reward for every image.

6. `Repeat the find mistakes -> retrain process once more.` Run a new 50,000-step policy with the round-1 reward model from step 5, collect another 200 episodes, and combine the two online datasets. Initialize from the round-1 vision adapter and head, then fine-tune for five more epochs with lower learning rates and greater emphasis on successful states. The resulting round-2 checkpoint is `frozen` and used as the reward model in the final online RL experiment.

#### **Results**

The selected refined PickCube reward model achieved:

| Split | Overall MAE ↓ | Offline MAE ↓ | Online MAE ↓ | Macro-bin MAE ↓ | Low-reward false positives ↓ |
|---|---:|---:|---:|---:|---:|
| Validation | 0.0351 | 0.0540 | 0.0253 | 0.1363 | 0.251% |
| Test | 0.0309 | 0.0484 | 0.0215 | 0.1181 | 0.305% |

The low online MAE and `0.305%` test false-positive rate show that policy-generated low-reward states were rarely assigned an exploitable high reward.

**Result file:** [`results/pickcube_reward_head_pretraining_evaluation.json`](results/pickcube_reward_head_pretraining_evaluation.json)


#### **Code Map**

| Step | Code |
|---|---|
| Generate normalized-reward trajectories | [`generate_reward_data.py`](src/vlm_reward/data/generate_reward_data.py) |
| Build paired reward JSONL | [`build_reward_dataset.py`](src/vlm_reward/data/build_reward_dataset.py) |
| Reward-contrastive dataset and training | [`reward_dataset.py`](src/vlm_reward/training/reward_dataset.py), [`train_reward_contrastive.py`](src/vlm_reward/training/train_reward_contrastive.py) |
| Attention pooler and MLP | [`vision.py`](src/vlm_reward/models/vision.py), [`reward.py`](src/vlm_reward/models/reward.py) |
| Frozen-head training | [`train_reward_head.py`](src/vlm_reward/training/train_reward_head.py) |
| Online false-positive collection | [`collect_hard_negatives.py`](src/vlm_reward/rl/collect_hard_negatives.py) |
| Joint encoder/head refinement | [`train_reward_head_joint.py`](src/vlm_reward/training/train_reward_head_joint.py) |
| Batched inference used by RL | [`reward_predictor.py`](src/vlm_reward/models/reward_predictor.py) |

#### **Run: Offline Reward Data**

```bash
vlm-generate-reward-data \
  --data-root data/reward_contrastive \
  --demo-root demos \
  --camera-modes static moving_mounted \
  --tasks PickCube-v1 PushCube-v1 PegInsertionSide-v1 \
  --view topview \
  --num-rollouts 500 \
  --num-frames 20 \
  --split-percentages 0.20 0.25 0.35 0.20 \
  --render-control-mode pd_ee_delta_pos \
  --seed 42 \
  --deterministic
```

Build the all-task dataset and the PickCube-only head dataset:

```bash
vlm-build-reward-data \
  --data-root data/reward_contrastive \
  --output-root finetune_data/reward_contrastive \
  --secondary-camera-mode moving_mounted \
  --tasks PickCube-v1 PushCube-v1 PegInsertionSide-v1 \
  --levels expert partial random regressing \
  --view topview \
  --num-frames 20 \
  --train-fraction 0.80 \
  --val-fraction 0.10 \
  --train-secondary-fraction 0.70 \
  --tiny-per-task 2 \
  --seed 42

vlm-build-reward-data \
  --data-root data/reward_contrastive \
  --output-root finetune_data/reward_head_pickcube \
  --secondary-camera-mode moving_mounted \
  --tasks PickCube-v1 \
  --levels expert partial random regressing \
  --view topview \
  --num-frames 20 \
  --train-fraction 0.80 \
  --val-fraction 0.10 \
  --train-secondary-fraction 0.70 \
  --tiny-per-task 2 \
  --seed 42
```

Train the reward-contrastive adapter:

```bash
vlm-train-reward-contrastive \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --train-jsonl finetune_data/reward_contrastive/train.jsonl \
  --val-jsonl finetune_data/reward_contrastive/val.jsonl \
  --output-dir finetuning_output/Qwen3-VL-8B-Reward-Contrastive/lora_weights \
  --num-train-epochs 1 \
  --learning-rate 1e-4 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --save-strategy steps \
  --save-steps 25 \
  --eval-strategy steps \
  --eval-steps 100 \
  --dataloader-num-workers 0 \
  --dataloader-pin-memory \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --contrastive-weight 1.0 \
  --seed 42 \
  --deterministic \
  --bf16 \
  --qlora
```

Optional autoregressive reward diagnostic:

```bash
vlm-infer-reward \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --adapter-dir finetuning_output/Qwen3-VL-8B-Reward-Contrastive/lora_weights \
  --jsonl finetune_data/reward_contrastive/test.jsonl \
  --output-json finetuning_output/Qwen3-VL-8B-Reward-Contrastive/reward_predictions.json \
  --max-new-tokens 512 \
  --batch-size 4 \
  --dataloader-num-workers 4 \
  --bf16

vlm-score-reward \
  --input-json finetuning_output/Qwen3-VL-8B-Reward-Contrastive/reward_predictions.json \
  --output-json finetuning_output/Qwen3-VL-8B-Reward-Contrastive/reward_metrics.json
```

This checks generated reward text. Online RL uses the direct MLP head below.

Train the balanced frozen head. The encoder is run once to build an embedding cache; the 300-epoch head sweep is then fast.

```bash
vlm-train-reward-head \
  --task PickCube-v1 \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --adapter-dir finetuning_output/Qwen3-VL-8B-Reward-Contrastive/lora_weights \
  --train-jsonl finetune_data/reward_head_pickcube/train.jsonl \
  --val-jsonl finetune_data/reward_head_pickcube/val.jsonl \
  --test-jsonl finetune_data/reward_head_pickcube/test.jsonl \
  --output-dir finetuning_output/Qwen3-VL-8B-Reward-Head-PickCube \
  --embedding-batch-size 32 \
  --head-batch-size 4096 \
  --num-workers 8 \
  --hidden-dim 256 \
  --dropout 0.1 \
  --epochs 300 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --balance-exponents 0.4 0.5 0.6 \
  --macro-selection-weight 1.0 \
  --seed 42 \
  --deterministic \
  --use-both-views \
  --bf16 \
  --device cuda
```

The selected checkpoint is:

```text
finetuning_output/Qwen3-VL-8B-Reward-Head-PickCube/best_partial_balanced_reward_head.pth
```

#### **Run: Online Refinement Round 1**

Generate two probe actors with the frozen head:

```bash
for seed in 42 43; do
  vlm-train-rl \
    --task PickCube-v1 \
    --max-steps 60000 \
    --batch-size 1024 \
    --learning-starts 5000 \
    --random-steps 5000 \
    --seed "$seed" \
    --eval-freq 10000 \
    --eval-episodes 5 \
    --save-dir "finetuning_output/pickcube_reward_probe_seed${seed}/weights" \
    --reward-head-checkpoint finetuning_output/Qwen3-VL-8B-Reward-Head-PickCube/best_partial_balanced_reward_head.pth \
    --annotation-episodes 8 \
    --reward-head-batch-size 512 \
    --env-success-override \
    --bf16 \
    --device cuda
done
```

Collect 300 online episodes across the two actors:

```bash
vlm-collect-hard-negatives \
  --task PickCube-v1 \
  --actor-checkpoints \
    finetuning_output/pickcube_reward_probe_seed42/weights/PickCube-v1/final_actor_export.pth \
    finetuning_output/pickcube_reward_probe_seed43/weights/PickCube-v1/final_actor_export.pth \
  --reward-head-checkpoint finetuning_output/Qwen3-VL-8B-Reward-Head-PickCube/best_partial_balanced_reward_head.pth \
  --output-dir finetune_data/reward_head_online_pickcube \
  --episodes-per-checkpoint 150 \
  --prediction-batch-size 512 \
  --episodes-per-prediction-batch 10 \
  --validation-fraction 0.15 \
  --test-fraction 0.15 \
  --jpeg-quality 95 \
  --seed 42 \
  --bf16 \
  --device cuda
```

Jointly refine vision LoRA, pooler, and head:

```bash
vlm-train-reward-head-joint \
  --task PickCube-v1 \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --adapter-dir finetuning_output/Qwen3-VL-8B-Reward-Contrastive/lora_weights \
  --reward-head-checkpoint finetuning_output/Qwen3-VL-8B-Reward-Head-PickCube/best_partial_balanced_reward_head.pth \
  --offline-train-jsonl finetune_data/reward_head_pickcube/train.jsonl \
  --offline-val-jsonl finetune_data/reward_head_pickcube/val.jsonl \
  --offline-test-jsonl finetune_data/reward_head_pickcube/test.jsonl \
  --online-train-jsonl finetune_data/reward_head_online_pickcube/train.jsonl \
  --online-val-jsonl finetune_data/reward_head_online_pickcube/val.jsonl \
  --online-test-jsonl finetune_data/reward_head_online_pickcube/test.jsonl \
  --output-dir finetuning_output/Qwen3-VL-8B-Reward-Head-PickCube-Joint \
  --epochs 5 \
  --batch-size 16 \
  --gradient-accumulation-steps 4 \
  --vision-learning-rate 1e-5 \
  --head-learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --balance-exponent 0.5 \
  --online-weight 1.5 \
  --hard-negative-weight 2.0 \
  --success-selection-weight 1.0 \
  --false-positive-selection-weight 0.25 \
  --minimum-prediction-std 1e-4 \
  --max-grad-norm 1.0 \
  --num-workers 8 \
  --seed 42 \
  --deterministic \
  --bf16 \
  --device cuda
```

#### **Run: Online Refinement Round 2**

Run a 50k-step policy with the round-1 head to expose its remaining errors:

```bash
vlm-train-rl \
  --task PickCube-v1 \
  --max-steps 50000 \
  --seed 44 \
  --eval-freq 5000 \
  --eval-episodes 10 \
  --save-dir finetuning_output/pickcube_reward_exploit_round2/weights \
  --reward-head-checkpoint finetuning_output/Qwen3-VL-8B-Reward-Head-PickCube-Joint/best_reward_head_joint.pth \
  --annotation-episodes 16 \
  --reward-head-batch-size 512 \
  --env-success-override \
  --bf16 \
  --device cuda
```

Collect its transitions with the round-1 head:

```bash
vlm-collect-hard-negatives \
  --task PickCube-v1 \
  --actor-checkpoints finetuning_output/pickcube_reward_exploit_round2/weights/PickCube-v1/final_actor_export.pth \
  --reward-head-checkpoint finetuning_output/Qwen3-VL-8B-Reward-Head-PickCube-Joint/best_reward_head_joint.pth \
  --output-dir finetune_data/reward_head_online_pickcube_round2 \
  --episodes-per-checkpoint 200 \
  --prediction-batch-size 512 \
  --episodes-per-prediction-batch 10 \
  --validation-fraction 0.15 \
  --test-fraction 0.15 \
  --jpeg-quality 95 \
  --seed 4242 \
  --bf16 \
  --device cuda
```

Combine both rounds without mixing train, validation, and test splits:

```bash
mkdir -p finetune_data/reward_head_online_pickcube_combined

for split in train val test; do
  cat \
    "finetune_data/reward_head_online_pickcube/${split}.jsonl" \
    "finetune_data/reward_head_online_pickcube_round2/${split}.jsonl" \
    > "finetune_data/reward_head_online_pickcube_combined/${split}.jsonl"
done
```

Train the final round-2 model:

```bash
vlm-train-reward-head-joint \
  --task PickCube-v1 \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --adapter-dir finetuning_output/Qwen3-VL-8B-Reward-Head-PickCube-Joint/vision_adapter \
  --reward-head-checkpoint finetuning_output/Qwen3-VL-8B-Reward-Head-PickCube-Joint/best_reward_head_joint.pth \
  --offline-train-jsonl finetune_data/reward_head_pickcube/train.jsonl \
  --offline-val-jsonl finetune_data/reward_head_pickcube/val.jsonl \
  --offline-test-jsonl finetune_data/reward_head_pickcube/test.jsonl \
  --online-train-jsonl finetune_data/reward_head_online_pickcube_combined/train.jsonl \
  --online-val-jsonl finetune_data/reward_head_online_pickcube_combined/val.jsonl \
  --online-test-jsonl finetune_data/reward_head_online_pickcube_combined/test.jsonl \
  --output-dir finetuning_output/Qwen3-VL-8B-Reward-Head-PickCube-Joint-Round2 \
  --epochs 5 \
  --batch-size 16 \
  --gradient-accumulation-steps 4 \
  --vision-learning-rate 5e-6 \
  --head-learning-rate 2e-5 \
  --weight-decay 1e-4 \
  --balance-exponent 0.5 \
  --online-weight 1.5 \
  --hard-negative-weight 2.0 \
  --success-selection-weight 1.5 \
  --false-positive-selection-weight 0.25 \
  --minimum-prediction-std 1e-4 \
  --max-grad-norm 1.0 \
  --num-workers 8 \
  --seed 42 \
  --deterministic \
  --bf16 \
  --device cuda
```

The final reward checkpoint is:

```text
finetuning_output/Qwen3-VL-8B-Reward-Head-PickCube-Joint-Round2/best_reward_head_joint.pth
```

### **3.2 Online RL with the Learned Reward**

#### **Method**

The actor and critic operate on ManiSkill state observations, not images. Images are rendered only for the frozen reward model, which maps each frame through the adapted Qwen vision encoder, attention pooler, and MLP head to obtain a scalar reward. The policy is therefore trained with visual rewards even though its action network uses simulator states.

`Why start with a bootstrap actor at 50% success rate?` A random PickCube policy almost never grasps, lifts, or places the cube. Training from scratch therefore fills replay with nearly identical low-progress states and gives the actor little evidence of which actions lead toward success. Before a random policy discovers useful manipulation behavior, it may find visual states that the reward model incorrectly scores highly. RL then reinforces these mistakes instead of learning the task, causing reward hacking. Our from-scratch visual-reward runs remained at 0% success.

We address this exploration problem by first `training an actor with ManiSkill's normalized dense reward`. Training stops at the first 20-episode evaluation with at least 40% success; the saved actor reached `50%` in that evaluation and has already visited grasping, lifting, and near-goal states. Consequently, the final experiment measures whether the learned visual reward can `improve a partially competent policy`, not whether it can learn PickCube from scratch. The environment itself is unchanged, i.e. the cube and goal are still initialized normally. Only the actor's starting weights are different. 

The final CrossQ run proceeds as follows:

1. `Initialize only the actor.` Load the bootstrap actor, verify its initial success rate, and save it as the step-zero baseline. The critics, optimizers, and replay buffer are created from scratch, so native-reward value estimates and transitions are not transferred into the visual-reward run.

2. `Prefill replay with competent behavior.` Run the bootstrap actor deterministically for 5,000 steps before learning starts. Unlike random exploration, this gives the new critics an initial dataset containing meaningful PickCube behavior.

3. `Annotate complete episodes with visual rewards.` Render each transition and process completed episodes in batches with the frozen round-2 Qwen vision encoder, attention pooler, and MLP. Autoregressive language generation is bypassed. 

4. `Preserve an unambiguous success target.` If ManiSkill reports task success, override the predicted reward with `1.0`. This terminal supervision prevents the learned model from assigning a weak reward to a genuine success; all non-success transitions still use the visual reward prediction.

5. `Update CrossQ from the annotated replay buffer.` The critics learn the return induced by the visual rewards, and the actor is optimized against those critics. A small KL penalty (weight 0.05) keeps the updated action distribution near the competent bootstrap actor, reducing rapid drift toward reward-model exploits while still allowing improvement.

6. `Stabilize early optimization.` Train the critics alone for the first 5,000 gradient updates, use a lower learning rate for the actor than for the critics, and fix the SAC entropy temperature at `0.003`.

Every 1,250 environment steps, the deterministic actor is tested on 20 simulator episodes generated with fixed evaluation seeds. Success is measured using ManiSkill's environment success flag, and evaluation episodes are never added to replay. For example, 80% training success means that 16 of these 20 episodes succeeded. Final reporting evaluates the initial and best actors on 200 different held-out episodes.

#### **Results**

| Evaluation | Initial actor | Best actor | Improvement |
|---|---:|---:|---:|
| Training evaluation, 20 fixed-seed episodes | 50% | **80%** at step 116,524 | +30 percentage points |
| Held-out evaluation, 200 different episodes | 60.5% | **77.0%** | +16.5 percentage points |

The `80%` value is the peak recurring training evaluation: `16 of 20 episodes` succeeded. On the 200-episode held-out evaluation, the same selected checkpoint succeeded in `154 episodes`, compared with `121` for the initial actor.

**Result files:** [`results/pickcube_online_rl_training.json`](results/pickcube_online_rl_training.json) and [`results/pickcube_online_rl_heldout.json`](results/pickcube_online_rl_heldout.json)

#### **Code Map**

| Step | Code |
|---|---|
| Training orchestration | [`train.py`](src/vlm_reward/rl/train.py) |
| Actor, critic, batch renormalization, replay | [`core.py`](src/vlm_reward/rl/core.py) |
| CrossQ/SAC updates and KL anchor | [`updates.py`](src/vlm_reward/rl/updates.py) |
| Environment state and rendering | [`environments.py`](src/vlm_reward/rl/environments.py) |
| Batched visual reward annotation | [`rewards.py`](src/vlm_reward/rl/rewards.py) |
| Checkpoints and resume state | [`checkpoints.py`](src/vlm_reward/rl/checkpoints.py), [`training_state.py`](src/vlm_reward/rl/training_state.py) |
| Fixed-seed policy evaluation | [`evaluation.py`](src/vlm_reward/rl/evaluation.py), [`evaluate_checkpoints.py`](src/vlm_reward/rl/evaluate_checkpoints.py) |

#### **Run**

Create the bootstrap actor:

```bash
vlm-train-rl \
  --task PickCube-v1 \
  --max-steps 150000 \
  --batch-size 1024 \
  --utd-ratio 1 \
  --learning-starts 5000 \
  --random-steps 5000 \
  --seed 42 \
  --deterministic \
  --eval-freq 2500 \
  --eval-episodes 20 \
  --eval-seed 100000 \
  --target-success-rate 0.95 \
  --bootstrap-target-success-rate 0.40 \
  --save-dir finetuning_output/pickcube_native_bootstrap_40/weights \
  --save-eval-snapshots \
  --use-env-rewards \
  --device cuda \
  --actor-hidden-dim 256 \
  --critic-hidden-dim 256 \
  --adam-beta1 0.5 \
  --learning-rate 3e-4 \
  --gamma 0.8 \
  --bootstrap-at-done always \
  --buffer-size 1000000 \
  --policy-delay 3 \
  --batch-renorm-warmup-steps 12000 \
  --actor-max-grad-norm 1.0
```

Training stops at the first evaluation at or above 40% and writes `bootstrap_actor.pth`.

Run the final visual-reward experiment:

```bash
vlm-train-rl \
  --task PickCube-v1 \
  --max-steps 150000 \
  --batch-size 1024 \
  --utd-ratio 1 \
  --learning-starts 5000 \
  --random-steps 0 \
  --deterministic-prefill-steps 5000 \
  --seed 44 \
  --deterministic \
  --eval-freq 1250 \
  --eval-episodes 20 \
  --eval-seed 100000 \
  --target-success-rate 0.90 \
  --minimum-steps-before-convergence 30000 \
  --save-dir finetuning_output/pickcube_vlm_fullreward_relaxed_kl_seed44/weights \
  --save-eval-snapshots \
  --reward-head-checkpoint finetuning_output/Qwen3-VL-8B-Reward-Head-PickCube-Joint-Round2/best_reward_head_joint.pth \
  --annotation-episodes 16 \
  --reward-scale 1.0 \
  --reward-head-batch-size 256 \
  --env-success-override \
  --bf16 \
  --device cuda \
  --init-actor-checkpoint finetuning_output/pickcube_native_bootstrap_40/weights/PickCube-v1/bootstrap_actor.pth \
  --reference-kl-weight 0.05 \
  --no-restore-alpha-from-init \
  --minimum-initial-success-rate 0.45 \
  --initial-eval-episodes 20 \
  --actor-hidden-dim 256 \
  --critic-hidden-dim 256 \
  --adam-beta1 0.5 \
  --learning-rate 3e-4 \
  --actor-learning-rate 7.5e-5 \
  --gamma 0.8 \
  --alpha 0.003 \
  --bootstrap-at-done always \
  --buffer-size 1000000 \
  --policy-delay 3 \
  --critic-only-warmup-updates 5000 \
  --batch-renorm-warmup-steps 12000 \
  --actor-max-grad-norm 1.0
```

Evaluate the initial and best actors on 200 held-out episodes:

```bash
vlm-evaluate-actors \
  finetuning_output/pickcube_vlm_fullreward_relaxed_kl_seed44/weights/PickCube-v1/initial_actor_export.pth \
  finetuning_output/pickcube_vlm_fullreward_relaxed_kl_seed44/weights/PickCube-v1/best_actor_export.pth \
  --episodes 200 \
  --base-seed 300000 \
  --device cuda \
  --output finetuning_output/pickcube_vlm_fullreward_relaxed_kl_seed44/heldout_eval_200_seed300000.json
```

## **Historical Progress-as-Reward Trial**

We also include the legacy code in which we used progress as rewards for online RL in this trajectory. 

```bash
python src/historical_progress_reward/generate_maniskill3_rl_trajectories.py

python src/historical_progress_reward/cleanrl_crossq.py \
  --task PickCube-v1 \
  --adapter-dir finetuning_output/Qwen3-VL-8B-Progress-Contrastive/lora_weights \
  --moving-mounted-camera \
  --vlm-context-len 20 \
  --vlm-batch-size 1 \
  --max-steps 300000 \
  --batch-size 1024 \
  --utd-ratio 1 \
  --learning-starts 5000 \
  --eval-freq 5000 \
  --eval-episodes 20 \
  --bf16 \
  --4bit-quant \
  --device cuda \
  --save-dir finetuning_output/historical_progress_reward/weights
```

## **Project Structure**

```text
.
├── configs/
│   ├── baselines/paper_reproduction.yaml   # API reproduction protocol
│   └── configs.yaml                        # tasks and shared data defaults
├── demos/                                  # source ManiSkill demonstrations
│   └── <task>/motionplanning/
├── Documents/                              # project references and proposal
├── results/                                # compact reported metrics
├── src/
│   ├── vlm_reward/                         # maintained package
│   │   ├── baselines/                      # zero/few-shot API reproduction
│   │   ├── data/                           # trajectory and JSONL generation
│   │   ├── evaluation/                     # progress/reward inference metrics
│   │   ├── models/                         # Qwen, contrastive, pooler, reward head
│   │   ├── rl/                             # CrossQ training and evaluation
│   │   ├── training/                       # datasets and fine-tuning commands
│   │   └── runtime.py                      # seeds and device helpers
│   └── historical_progress_reward/         # collaborator's retained trial
├── data/                                   # generated images; ignored by Git
├── finetune_data/                          # generated JSONL datasets; ignored
├── finetuning_output/                      # checkpoints and logs; ignored
├── output/                                 # API baseline outputs; ignored
├── requirements.txt                        # pip dependencies
└── pyproject.toml                          # package and command entry points
```

Use `--overwrite` only when intentionally rebuilding a generated dataset. Training commands refuse to mix a new run with an existing output directory.
