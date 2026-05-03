RoboReward: Full rollout as input, reward only at the end?

`Meta-World`: Meta-World is a simulated robotic manipulation benchmark with 50 distinct Sawyer robot tasks, such as reaching, pushing, pick-and-place, drawer opening, button pressing, door opening, window opening, and peg insertion. Meta-World gives you a controlled set of manipulation tasks where ground-truth rewards and success metrics already exist. That makes it very useful for evaluating whether your VLM-generated rewards are actually meaningful. We can compare $r_t^{VLM}$ with $r_t^{env}$, and also compare the final policy success rate.

`ManiSkill3`: a simulation benchmark for robotic object manipulation tasks with a diverse range of environments, robot types (humanoids, mobile manipulators, and single-arm robots) and with realistic physical simulations (Sim2Real is possible).

Can be used to experiment generalization across visual domains

ManiSill3 tasks:

- Lift Cube
- Turn Faucet
- Open Cabinet Door
- Open Cabinet Drawer
- Push Chair
- Stack Cube

`LSceneLLM`: Not very relevant, but LSceneLLM supports the idea that task-relevant visual focus matters. The paper shows that large scene understanding improves when the model adaptively selects relevant regions instead of processing all visual tokens equally. e.g. We can implement full image reward vs cropped region reward. Or maybe help the robot when the movement is really subtle (same idea but 2D).
