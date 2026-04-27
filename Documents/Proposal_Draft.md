## **Title**

Vision-Language Models as Pairwise Progress Rewards for Visual Robotic Reinforcement Learning

## **Motivation**

Reinforcement learning often requires either manually specified reward functions for learned reward models from human feedback, both of which can be costly or difficult to obtain [Roc24]. In robotic manipulation, reward design is especially problematic because task progress may depend on task-specific quantities such as object position, gripper-object distance, or goal completion, which are not always available from raw visual observations alone [...].

Recent work has shown that pretrained vision-language models can be used as zero-shot reward models by scoring visual observations against natural-language task descriptions [Roc24]. Rocamonde et al. implement this idea using CLIP, where the reward is computed from the cosine similarity between an image embedding and a text-goal embedding [Roc24]. However, single-frame image-text similarity may not be sufficient for robotic manipulation tasks where progress is relative, for example when comparing whether a drawer is more open than before or whether an object has moved closer to a target [...].

This project investigates a pairwise alternative: Instead of asking whether one image looks like the goal, we ask whether a later frame shows progress compared to an earlier frame. The project topic explicitly proposes presenting a VLM with before/after frame pairs and goal text to produce progress judgments that can be used for ranking rollouts and training an RL agent.

## **Research Question**

The main research question is:

Can a frozen vision-language model judge manipulation task progress from before/after image pairs well enough to provide a useful reward signal for reinforcement learning?

More specifically, this project asks:

1. Can pairwise VLM rewards correlate with Meta-World ground-truth rewards and success metrics?
2. Can pairwise VLM rewards rank successful rollouts above failed or partially successful rollouts?
3. Can an RL agent trained with pairwise VLM rewards improve task success compared to baseline reward signals?
4. Where do VLM judgments break down, such as visually similar states, subtle differences, long-horizon tasks, or complex tasks?

## **Proposed Method**

We propose a pairwise VLM progress reward. Given two rendered frames from a rollout, $I_t$ and $I_{t + k}$, and a language goal $g$, the VLM predicts whether the second frame shows progress toward the goal compared to the first frame. The proposed reward is:

$$
r_t^{VLM} = VLMProgress(I_t, I_{t + k}, g)
$$

The VLM output will be converted into a numerical reward using the scale:

$$
r_t^{VLM} ∈ {1, 2, 3, 4, 5}
$$

The reward format is a discrete progress score, motivated by RoboReward's use of 1-5 progress labels and its finding that progress-based rewards are more useful than binary success rewards in their RL experiments [Lee26]. 

This pairwise reward will be compared against a single-frame CLIP reward:

$$
r_t^{CLIP} = cos(CLIP_{image}(I_t), CLIP_{text}(g))
$$

## **Experimental Setup**

The project will use Meta-World as the simulation benchmark. Meta-World contains 50 simulated manipulation tasks and uses a shared Sawyer robot setup across tasks [Yu21]. The benchmark defines task success metrics, often based on the distance between a task-relevant object and its target pose [Yu21].

The initial task set will be:

1. 
2. 
3. 
4. 
5. 

For each task, rollout will be collected from:

1.
2. 
3. 
4.

Each rollout consists of a sequence of states and actions:

$$
τ = (s_0, a_0, s_1, a_1, ..., s_T)
$$

Each state will be rendered as an RGB image for VLM reward compotation.

## **Reward Baselines**

The project will compare the following reward signals:

1. Meta-World ground-truth reward: Meta-World provides task reward functions and success metrics for its manipulation environments [Yu21]. This reward will serve as the reference or upper baseline.

2. Single-frame CLIP reward: The CLIP baseline computes image-text similarity between the current frame and the language goal [Roc24].

3. Pairwise VLM progress reward: The proposed method uses two frames and a goal instruction to judge progress between the earlier and later frame.

## **RL Training Plan**

THe second stage of the project will train a RL agent using the VLM-generated reward. The RL algorithm will be SAC. During RL training, the environment transition remains:

$$
s_t, a_t → s_{t + 1}
$$

The reward used for learning is replaced by:

$$
r_t^{env} → r_t^{VLM}
$$

The policy will therefore learn from transitions of the form:

$$
(s_t, a_t, r_t^{VLM}, s_{t + 1})
$$

The exact strategy for computing VLM rewards during training will be: ... (When do we call the VLM to give us the reward, because calling it every step will make the simulation very slow? Should we query every $k$ step instead of every step, i.e. compare $I_t$ and $I_{t + k}$, and try different values of $k$? Or other ideas?)

## **Evaluation**

The project will use two levels of evaluation.

### **Offline Reward Evaluation**

First, the reward signals will be evaluated before full RL training. The offline metrics will be:

- Reward correlation with Meta-World reward
- MAE against binned progress labels
- Rollout ranking accuracy
- Success/failure classification AUC

### **RL Policy Evaluation**

After training, policies will be evaluated using the true Meta-World success metrics rather than the VLM score. (Meta-World defines interpretable task success metrics for its task, more detail here please).

The main policy-level metric will be: ... .

## **Expected Contribution**

This project will evaluate whether pairwise VLM process judgements can serve as reward signals for visual robotic RL. The expected outputs are:

1. A pipeline for scoring Meta-World frame pairs with a VLM-based progress reward.
2. A comparison between Meta-World rewards, CLIP-style rewards, and pairwise VLM rewards.
3. Offline reward-quality analysis.
4. RL training results using SAC as the training algorithm.
5. Failure analysis on cases such as visually similar states, subtle differences, long-horizon tasks, and complex tasks, which are open questions in the project description.