`Problem motivation`: In visual robotic RL, designing reward functions is hard. For every task, humans usually need to define what counts as progress or success. Example: For a drawer-opening task, the reward could depend on drawer position. But this requires simulator state access or hand-written task-specific logic.

`Main research question`: Can we replace hand-designed rewards with VLM judgments from images?

The core idea is to use a VLM as a zero-shot reward model. Instead of giving the RL agent a manually coded reward, you show the VLM pairs of frames from a rollout: $(s_t, s_{t + k})$, plus the goal text: “Open the drawer.” The VLM judges whether the second frame is closer to the goal than the first frame. This gives a reward-like signal: $r_t = VLMProgress(s_t , s_{t + k}, goal)$. Then this reward can be used to rank rollouts, label progress, or train an RL policy.

RoboReward: Full rollout as input, reward only at the end?

Project pipeline:

1. Choose benchmark tasks

Use a manipulation benchmark, probably Meta-World, with tasks like:

- push object
- open drawer
- reach target
- pick and place
- button press

Include also longer horizon, multi stage tasks??

`Meta-World`: Meta-World is a simulated robotic manipulation benchmark with 50 distinct Sawyer robot tasks, such as reaching, pushing, pick-and-place, drawer opening, button pressing, door opening, window opening, and peg insertion.

1. Collect rollouts

Run policies with different success rates:

- random policy
- partially trained policy
- expert / scripted policy, if available

This gives trajectories:

$$
s_0, a_0, s_1, a_1, ..., s_T
$$​

where each state can be rendered as an image.

`Meta-World`: Meta-World gives you the simulated robot, object states, actions, rewards, and success labels. The benchmark uses a simulated Sawyer robot, with a shared action space across tasks. So your rollouts become: $s_0, s_1, ..., s_T$, where $s_T$ can be rendered as an RGB image and passed to the VLM.

3. Query the VLM

For frame pairs, ask the VLM whether progress happened with some type of specific prompts.

`VLM-RM`: Use a pretrained vision-language model as a zero-shot reward model for RL. Their simplest implementation uses CLIP:

$$
R_{CLIP}(s) = cos(CLIP_{text}(goal), CLIP_{image}(image(s)))
$$

So the reward is high when the current image looks semantically close to the goal text. They use this reward instead of the environment reward and train RL agents with standard algorithms like DQN and SAC (2 standard RL algorithms).

Their reward is more simple: $r_t = VLM(s_t, goal)$, but the project idea is slightly different and potentially stronger: $r_t = VLMProgress(s_t, s_{t + k}, goal)$. That means the project compares before/after frame pairs, not just a single current frame. This is important because manipulation progress is often relative.

Potential reuse:
- VLM-RM can become the `baseline method` (reuse the simplest VLM reward baseline $R_{CLIP}(s) = cos(CLIP_{text}(goal), CLIP_{image}(image(s)))$).

- They improve CLIP reward by adding a baseline prompt, e.g.: Goal: “a humanoid robot kneeling”, baseline: “a humanoid robot”. Then they project the image embedding toward the direction from baseline to goal. Intuitively, this removes irrelevant visual information and focuses the reward on the goal-relevant change.

- Failure analysis: They found failures when visual differences are subtle, when environments look unrealistic, or when the VLM lacks spatial reasoning. For example, CLIP failed on some humanoid poses like “arms crossed” and “standing on one leg.”

Example in proposal: Rocamonde et al. show that pretrained VLMs can serve as zero-shot reward models by scoring the similarity between visual observations and natural language task descriptions. Their results demonstrate that CLIP-based rewards can train RL agents without manually engineered reward functions, but also reveal limitations for subtle visual states and off-distribution renderings. Building on this, our project studies whether pairwise before/after VLM judgments can provide a more progress-sensitive reward signal for robotic manipulation tasks.

`LSceneLLM`: Not very relevant, but LSceneLLM supports the idea that task-relevant visual focus matters. The paper shows that large scene understanding improves when the model adaptively selects relevant regions instead of processing all visual tokens equally. e.g. We can implement full image reward vs cropped region reward. Or maybe help the robot when the movement is really subtle (same idea but 2D).

4. Convert VLM answers to rewards

- Ask VLM to output a point scoring 0-5
- Ask VLM to compare 2 frames and score if second frame is closer to objective, so output a yes or no (or use logits of yes/no token as reward?)
- Add regression head

5. Evaluate the VLM reward

You compare VLM judgments against ground truth environment rewards.

Important questions:

- Can the VLM correctly rank successful rollouts above failed ones?
- Does it detect small progress?
- Does it fail on visually subtle tasks?
- Does it generalize across different manipulation tasks? See in which task does the VLM fail?

`Meta-World`: Meta-World gives you a controlled set of manipulation tasks where ground-truth rewards and success metrics already exist. That makes it very useful for evaluating whether your VLM-generated rewards are actually meaningful. We can compare $r_t^{VLM}$ with $r_t^{env}$, and also compare the final policy success rate.

Meta-World lets us answer:

- Does the VLM reward correlate with the true task progress/success?
- Does training with the VLM reward improve actual Meta-World success rate?

Potential reward methods to compare:

- Original Meta-World reward (upper baseline)
- Random reward
- CLIP reward
- Robo Reward
- This project's reward
- Optional crop-based pairwise reward

Question: How do we compare the rewards? (MAE to ground truth rewards, EPIC distance to other rewards)

6. Use it for actual RL training

Prove the VLM works in a real agent case to improve task success rate. So the stronger version is not only “VLM can judge frames”, but:

An RL agent trained with VLM-derived rewards achieves higher success rate than a baseline.

Novelty:
- A wider variety of tasks + tasks with subtle changes in movement (e.g. very close to button but not touch it)
- Extend to multi-stage tasks?
- Fine-tuning possible?
- If model fails to detect subtle changes or struggles with spatial understanding we could explore the method proposed by LSceneLLM: Get 3D point cloud from MetaWorld and use LSceneLLM
