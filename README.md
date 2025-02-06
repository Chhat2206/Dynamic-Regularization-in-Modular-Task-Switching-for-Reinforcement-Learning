# Dynamic Regularization in Modular Task Switching for Reinforcement Learning

## Overview
This repository contains the implementation of dynamic regularization techniques for modular task switching environments using Deep Q-Networks (DQN). The project evaluates the impact of different regularization strategies—Cyclic, Structured, Randomized, and Standardized—on agent performance within the **CartPole** and **Acrobot** environments from OpenAI's Gymnasium.

The experiments are designed to test agent adaptability, knowledge retention, long-term learning stability, and resilience against noisy environments.

## Features
- **Dynamic Task-Specific Regularization:** Switches between Dropout, L1, L2, and Batch Normalization based on task requirements.
- **Modular Architecture:** Facilitates effective task isolation and knowledge retention.
- **Multi-Task Learning:** Supports cyclic, randomized, and structured task-switching.
- **Noise Resilience:** Evaluates agent performance under varying state and action noise conditions.
- **Custom Reward Shaping:** Optimizes learning efficiency for specific tasks like quick recovery, periodic swinging, and maintaining balance.

## Repository Structure
```
.
├── main-cartpole.py          # DQN implementation for the CartPole environment
├── main-acrobot.py           # DQN implementation for the Acrobot environment
├── Dynamic_Regularization_in_Modular_Task_Switching_RL.pdf # Research report detailing experimental methodology and results
├── README.md                 # Project documentation
```

## Environments
- **CartPole-v1:** Focuses on balancing a pole on a moving cart.
- **Acrobot-v1:** A two-link pendulum system that requires the agent to swing the end-effector to a target height.

## Regularization Techniques
- **Dropout:** Reduces overfitting by randomly dropping units during training.
- **L1 Regularization:** Promotes sparsity in network weights, enhancing robustness.
- **L2 Regularization:** Penalizes large weights to improve generalization.
- **Batch Normalization:** Normalizes layer inputs to stabilize learning.

## Task-Switching Modes
1. **Cyclic Task Switching:** Regularization techniques cycle through a predefined sequence.
2. **Structured Task Switching:** Retains memory of regularization strategies for specific tasks.
3. **Randomized Task Switching:** Randomly assigns regularization techniques per task.
4. **Standardized Regularization:** Applies a fixed regularization strategy throughout the training.

## Installation
Ensure you have Python 3.8+ installed. Then, run the following commands:

```bash
pip install gymnasium torch numpy matplotlib pandas
```

## How to Run
### CartPole Environment:
```bash
python main-cartpole.py
```

### Acrobot Environment:
```bash
python main-acrobot.py
```
