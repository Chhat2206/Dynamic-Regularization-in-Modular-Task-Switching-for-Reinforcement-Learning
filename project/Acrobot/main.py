#!pip install gymnasium
# !pip install tensorboard
# !rm -rf ./runs
# %load_ext tensorboard
# %tensorboard --logdir runs
import pickle
from matplotlib import pyplot as plt
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import datetime

# Create the environment
env = gym.make("Acrobot-v1")

# Set goals for custom task-switching
use_custom_goals = True  # Set False to train all episodes under the original goal in all episodes
is_rq1 = True  # Set this to True to evaluate RQ1, and automatically set RQ2 to False
is_rq2 = not is_rq1
mode = "none"  # Options are "none", "cyclic_task_switching", "randomized_task_switching"
goals = ["quick_recovery", "periodic_swing", "maintain_balance"]
fixed_reg_type = "dropout"  # Choose from ["dropout", "l1", "l2", "batch_norm"]

# Hyperparameters
learning_rate = 0.0001  # Slightly lower for more stable learning, OG: 0.00025 is REALLY BAD
gamma = 0.99
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995  # Slower decay from 0.995 to allow more exploration
min_epsilon = 0.01
batch_size = 128  # Recommended 128 or 256, prof says 32 has valuable research
replay_buffer_size = 1000000  # Increased to allow more diverse experiences
replay_buffer = deque(maxlen=replay_buffer_size)
num_episodes = 500  # Increased number of episodes, OG: 500
target_update_frequency = 500
performance_threshold = -100
stability_threshold = 0.01
window_size = 50
eval_interval = 50
# Training episodes with task switching
parameter_noise_stddev = 0.1  # Standard deviation for parameter noise (for RQ2)

# Timing variables
training_start_time = time.time()


# Reward shaping function
def shape_reward(state, next_state, reward):
    # Reward the agent for increasing the height of the tip of the second link
    if next_state[1] > state[1]:  # if the vertical position increases
        reward += 0.5  # Provide a small positive reward for progress
    return reward

# Neural network for DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, reg_type=None, num_modules=4):
        super(DQN, self).__init__()
        self.reg_type = reg_type
        self.num_modules = num_modules

        self.fc1 = nn.ModuleList([nn.Linear(input_dim, 128) for _ in range(num_modules)])
        self.bn1 = nn.ModuleList([nn.BatchNorm1d(128) for _ in range(num_modules)])
        self.fc2 = nn.Linear(128 * num_modules, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if len(x.shape) == 1: x = x.unsqueeze(0)  # Add a batch dimension, making it [1, input_dim]
        module_outputs = [torch.relu(layer(x)) for layer in self.fc1]

        if self.reg_type == "batch_norm":
            module_outputs = [
                self.bn1[i](output) if output.size(0) > 1 else output
                for i, output in enumerate(module_outputs)
            ]

        x = torch.cat(module_outputs, dim=-1)  # Concatenate outputs from all the modules
        if self.reg_type == "dropout": x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        if self.reg_type == "batch_norm" and x.size(0) > 1: x = self.bn2(x)
        return self.fc3(x)


# Function to assign rewards based on the goal
def get_goal_reward(reward, state, goal):
    if goal == "quick_recovery":
        if abs(state[0]) > 1.0:
            return reward + 1.0  # Reward the agent for quickly moving to a stable position from an extreme angle
        elif goal == "periodic_swing":
            return reward + np.clip(np.sin(state[1] * 5.0), -1, 1)  # Reward swinging behavior
    elif goal == "maintain_balance":
        return reward + (1.0 - abs(state[0]))  # Reward maintaining the top link at a horizontal position
    return reward

# Function to add noise to state
def add_noise_to_state(state, noise_level=0.1):
    noise = np.random.normal(0, noise_level, size=state.shape)
    noisy_state = state + noise
    return np.clip(noisy_state, env.observation_space.low, env.observation_space.high)


# Function to train the DQN using replay buffer
def train_dqn(q_network, target_network, optimizer, reg_type):
    if len(replay_buffer) < batch_size:
        #     print(f"Insufficient samples in replay buffer ({len(replay_buffer)}). Waiting to collect more samples.")
        return

    # Sample random mini-batch
    indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
    batch = [replay_buffer[i] for i in indices]
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

    # Compute the Q-values for current states
    current_q_values = q_network(states).gather(1, actions)
    assert current_q_values.requires_grad, "Q-values must require gradients for backward pass."

    # Compute the target Q-values
    with torch.no_grad():
        next_q_values = target_network(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * gamma * next_q_values

    # Compute loss
    loss = nn.MSELoss()(current_q_values, target_q_values)

    # Add L1 regularization if applicable
    if reg_type == "l1":
        l1_loss = sum(torch.abs(param).sum() for param in q_network.parameters())
        loss += 1e-4 * l1_loss

    # Update the network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Initialize Q-network and target network
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

# Define the available regularization techniques
regularization_types = ["dropout", "l1", "l2", "batch_norm"]

# Initialize variables for retention results and previous goals
previous_goals = []  # Stores previously learned goals to evaluate knowledge retention

results = {
    reg_type: {
        "rewards": [],
        "convergence_speed": [],
        "retention_score": [],
        "noise_resilience": [],
        "task_rewards": {goal: [] for goal in goals + ["original"]},
    }
    for reg_type in regularization_types
}

retention_results = {
    reg_type: {
        "retention_scores": [],
        "previous_goal_rewards": {},
        "long_term_adaptability": {}  # Add this to store the long-term adaptability scores for each regularization type
    } for reg_type in regularization_types
}

# --- Training DQN ---
# Define device (use GPU if available, otherwise fallback to CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training with device: {device}")

action_distributions = {}
possible_actions = [0, 1, 2]

# Evaluate agent on the given goal/environment
def evaluate_agent(env, q_network, num_episodes=5, goal="original"):
    total_rewards = []
    action_distribution = [0] * env.action_space.n  # Initialize action count for each action

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        total_reward = 0
        done = False
        step_count = 0
        max_eval_steps = 500

        while not done and step_count < max_eval_steps:
            with torch.no_grad():
                action = select_action(state.cpu().numpy(), epsilon=0, q_network=q_network)
                next_state, reward, done, _, _ = env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
                total_reward += reward
                action_distribution[action] += 1  # Increment action count
                state = next_state
                step_count += 1

        total_rewards.append(total_reward)
        print(f"Evaluation Episode: {episode + 1}, Goal: {goal}, Total Reward: {total_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"Average Reward after Evaluation for Goal '{goal}': {avg_reward}")
    print(f"Action Distribution during Evaluation: {action_distribution}")

    action_distributions[goal] = action_distribution
    return avg_reward, action_distribution


def add_noise_to_action(action, num_actions, noise_probability=0.1):
    if random.random() < noise_probability:
        return random.choice(range(num_actions))  # Select a random action
    return action


# Function to select action using epsilon-greedy policy
def select_action(state, epsilon, q_network, noise_level=0.1, action_noise_prob=0.1):
    try:
        noisy_state = add_noise_to_state(state, noise_level) if noise_level > 0 else state
    except ValueError as e:
        print(f"Error in state noise addition: {e}. Falling back to original state.")
        noisy_state = state

    if random.random() < epsilon:
        action = env.action_space.sample()  # Exploration
    else:
        with torch.no_grad():
            noisy_state = torch.tensor(noisy_state, dtype=torch.float32).to(device)
            q_values = q_network(noisy_state)
            action = int(torch.argmax(q_values).item())  # Exploitation

    # Apply noise to the selected action
    return add_noise_to_action(action, env.action_space.n, action_noise_prob)


avg_rewards_per_100_episodes = []
original_phase_rewards = []
task_switching_phase_rewards = []

performance_log = {goal: {"rewards": [], "convergence_speed": []} for goal in goals}
performance_log["original"] = {"rewards": [], "convergence_speed": []}
task_switch_rewards = []
task_switch_count = 0
total_steps = 0

# Function to determine if agent has converged
def is_converged(rewards, threshold, window_size=10):
    if len(rewards) < window_size:
        return False
    return np.mean(rewards[-window_size:]) >= threshold

def add_parameter_noise(model, stddev=0.1):
    with torch.no_grad():  # We don't want to learn or change weights permanently here
        for param in model.parameters():
            if param.requires_grad:  # Only change weights that are trainable
                # Add a small random value (noise) to each weight
                noise = torch.normal(mean=0.0, std=stddev, size=param.size()).to(param.device)
                param.add_(noise)  # Apply the noise to the parameter


# Dictionary to track convergence speed for each task
convergence_log = {goal: [] for goal in goals + ["original"]}

# Number of episodes taken to converge after switching to a task
current_task_episode_count = {goal: 0 for goal in goals + ["original"]}

# Function to evaluate knowledge retention
def evaluate_knowledge_retention(agent, env, learned_tasks, num_episodes=5):
    retention_rewards = {}
    # Log retention trends over tasks
    retention_trends = []

    # Log retention trends for all tasks
    for trend in retention_trends:
        general_writer.add_scalar(f"Retention/{trend['task']}/Reward", trend["reward"], 0)

    for task in learned_tasks:
        avg_reward = evaluate_agent(env, q_network, num_episodes=5, goal=task)[0]
        retention_trends.append({"task": task, "reward": avg_reward})
        total_rewards = []
        print(f"Evaluating knowledge retention for task: {task}")
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = select_action(state, epsilon=0, q_network=agent)  # Greedy policy
                next_state, reward, done, _, _ = env.step(action)
                total_reward += reward
                state = next_state
            total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards)

        retention_rewards[task] = avg_reward
        print(f"Knowledge Retention - Task: {task}, Avg Reward: {avg_reward}")
    return retention_rewards

# Function to evaluate long-term adaptability
def evaluate_long_term_adaptability(agent, env, tasks, num_episodes=5, max_eval_steps=300, early_stop_threshold=-200):
    adaptability_rewards = {}
    for task in tasks:
        total_rewards = []
        print(f"Evaluating long-term adaptability for task: {task}")
        for episode in range(num_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)
            done = False
            total_reward = 0
            step_count = 0
            while not done and step_count < max_eval_steps:
                with torch.no_grad():
                    action = select_action(state.cpu().numpy(), epsilon=0, q_network=agent)  # Greedy policy
                next_state, reward, done, _, _ = env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32).to(device)

                total_reward += reward
                state = next_state
                step_count += 1

                # Early stop if the episode reaches a significant negative reward
                if total_reward < early_stop_threshold and step_count > max_eval_steps // 3:
                    print(f"Early stop in episode {episode + 1} for task {task} due to low progress.")
                    break

            total_rewards.append(total_reward)

        avg_reward = np.mean(total_rewards)
        adaptability_rewards[task] = avg_reward
        print(f"Long-Term Adaptability - Task: {task}, Avg Reward: {avg_reward}")

    return adaptability_rewards

# Validation check for mode and regularization types
if mode == "none" and fixed_reg_type not in ["dropout", "l1", "l2", "batch_norm"]:
    raise ValueError(
        "Invalid fixed_reg_type provided for mode 'none'. Please choose from ['dropout', 'l1', 'l2', 'batch_norm']")

# Define the function to validate single regularization type
def validate_single_reg_type(q_network, target_network, expected_reg_type):
    if q_network.reg_type != expected_reg_type or target_network.reg_type != expected_reg_type:
        raise ValueError(
            f"Multiple regularization types detected. Only '{expected_reg_type}' is allowed for mode 'none'.")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
print(f"\nTraining with Mode: {mode}")

# Initialize model, target, optimizer, and replay buffer
# In "none" mode, use the fixed regularization type for the entire experiment
if mode == "none":
    reg_type = fixed_reg_type
else:
    reg_type = None  # Placeholder to be set dynamically based on task-switching

# Initialize Q-network and target network with the selected regularization type
q_network = DQN(env.observation_space.shape[0], env.action_space.n, reg_type=reg_type).to(device)
target_network = DQN(env.observation_space.shape[0], env.action_space.n, reg_type=reg_type).to(device)

# Ensure all parameters of the model have gradients enabled
for name, param in q_network.named_parameters():
    assert param.requires_grad, f"Parameter {name} does not require gradients."

# In "none" mode, validate that both networks use the same regularization type
if mode == "none":
    validate_single_reg_type(q_network, target_network, reg_type)

optimizer = optim.Adam(q_network.parameters(), lr=learning_rate, weight_decay=1e-4 if reg_type == "l2" else 0)
replay_buffer = deque(maxlen=replay_buffer_size)

epsilon = 1.0
episode_rewards = []  # Track rewards per episode
task_to_reg = {}  # Dictionary to store the task for regularized task switching
previous_goal = None
epoch_details = []

log_dir = f"runs/{mode}_{'RQ1' if is_rq1 else 'RQ2'}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
writers = {reg_type: SummaryWriter(log_dir=f"{log_dir}/{reg_type}") for reg_type in regularization_types}
general_writer = SummaryWriter(log_dir=f"{log_dir}/general")

hyperparams = {
    "learning_rate": learning_rate,
    "gamma": gamma,
    "epsilon_decay": epsilon_decay,
    "batch_size": batch_size,
    "num_episodes": num_episodes,
}

def validate_agent(agent, env, validation_goals, num_episodes=5, timeout=30):
    validation_results = {}

    for goal in validation_goals:
        print(f"\nValidating for goal: {goal}")

        # Record start time for this goal validation
        start_time = time.time()

        # Set validation parameters based on the goal
        if goal == "stabilize_at_angle":
            def reward_shaping(state, reward):
                if 0.8 < state[0] < 1.2:  # Close to horizontal position
                    reward += 0.5
                return reward

            noise_level = 0.0
            action_noise_prob = 0.0

        elif goal == "low_state_noise":
            reward_shaping = None
            noise_level = 0.0
            action_noise_prob = 0.0

        elif goal == "high_state_noise":
            reward_shaping = None
            noise_level = 0.3
            action_noise_prob = 0.0

        elif goal == "noisy_stabilization":
            def reward_shaping(state, reward):
                if abs(state[0]) < 0.5:  # Close to balance
                    reward += 1.0
                return reward

            noise_level = 0.2
            action_noise_prob = 0.0

        elif goal == "high_action_noise":
            reward_shaping = None
            noise_level = 0.0  # No state noise
            action_noise_prob = 0.3  # High action noise

        elif goal == "noisy_swing_maximization":
            def reward_shaping(state, reward):
                if abs(state[1]) > 1.5:  # High swing amplitude
                    reward += 1.0
                return reward

            noise_level = 0.3
            action_noise_prob = 0.0

        else:
            print(f"Unknown validation goal: {goal}")
            continue

        # Run episodes for the validation goal
        total_rewards = []
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                # Check timeout every step during the episode
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout:
                    print(f"Validation for goal '{goal}' took too long ({elapsed_time:.2f}s), skipping.")
                    validation_results[goal] = None  # Mark the goal as skipped
                    return validation_results  # Exit early if timeout is reached

                # Add noise to state if applicable
                noisy_state = add_noise_to_state(state, noise_level) if noise_level > 0 else state
                action = select_action(noisy_state, epsilon=0, q_network=agent, noise_level=noise_level,
                                       action_noise_prob=action_noise_prob)
                next_state, reward, done, _, _ = env.step(action)

                # Apply reward shaping if defined
                if reward_shaping is not None:
                    reward = reward_shaping(state, reward)

                total_reward += reward
                state = next_state

            total_rewards.append(total_reward)

        avg_reward = np.mean(total_rewards)
        validation_results[goal] = avg_reward
        print(f"Validation - Goal: {goal}, Avg Reward: {avg_reward}")

    return validation_results


validation_goals = [
    "stabilize_at_angle",
    "low_state_noise",
    "high_state_noise",
    "noisy_stabilization",
    "noisy_swing_maximization",
]

# Calculate success rate as the percentage of episodes where total reward >= threshold.
def calculate_success_rate(rewards, threshold):
    successes = [1 for r in rewards if r >= threshold]
    return (sum(successes) / len(rewards)) * 100 if rewards else 0

assigned_regularizations = set()
task_to_reg = {}

tasks = [
    "original",        # Episodes 1-50
    "quick_recovery",  # Episodes 51-100
    "periodic_swing",  # Episodes 101-150
    "maintain_balance",# Episodes 151-200
    "original",        # Episodes 201-250
    "maintain_balance",# Episodes 251-300
    "quick_recovery",  # Episodes 301-350
    "original",        # Episodes 351-400
    "periodic_swing",  # Episodes 401-450
    "maintain_balance" # Episodes 451-500
]

# Function to determine current task based on episode number
def get_current_task(episode_number):
    task_index = (episode_number - 1) // 50  # Divide by 50 to get the task cycle index
    task_index = task_index % len(tasks)  # Ensure the index wraps around after the correct amount of episodes
    return tasks[task_index]

chosen_reg_types = []

def get_next_reg(episode, cycle_length=4):
    if len(chosen_reg_types) < cycle_length:
        # First 4 episodes
        remaining_types = [reg for reg in regularization_types if reg not in chosen_reg_types]
        reg_type = random.choice(remaining_types)
        chosen_reg_types.append(reg_type)
    else:
        # Cycle for after the 4 episodes
        cycle_index = (episode - cycle_length) % cycle_length
        reg_type = chosen_reg_types[cycle_index]

    return reg_type

validation_log = []
validation_results_history = []

rolling_avg_rewards = []
rolling_std_rewards = []

cumulative_rewards = 0
previous_cumulative_rewards = 0
convergence_speeds = []

# Check if the agent has converged
def check_convergence(task_rewards, performance_threshold, window_size):
    # Check if task_rewards meets the convergence criteria
    if len(task_rewards) >= window_size:
        # Compute the rolling average of the last `window_size` rewards
        rolling_rewards = task_rewards[-window_size:]
        avg_reward = np.mean(rolling_rewards)
        std_reward = np.std(rolling_rewards)

        # Check if the average reward is greater than the performance threshold
        # and if the standard deviation is below a threshold indicating stability
        is_converged = avg_reward >= performance_threshold and std_reward < 0.01

        # Return a flag indicating convergence and the countdown to convergence
        countdown = max(0, window_size - len(task_rewards))  # Countdown to full window size
        return is_converged, countdown

    # If not enough rewards to check, return False for convergence and an arbitrary countdown
    return False, None

# Initialize lists to track convergence and stabilization details
convergence_details = []
stabilization_details = []

# Function to check convergence based on stabilization (rolling standard deviation of rewards)
def check_stabilization(episode_rewards, window_size, stabilization_threshold=0.01):
    if len(episode_rewards) >= window_size:
        rolling_std_reward = np.std(episode_rewards[-window_size:])
        return rolling_std_reward < stabilization_threshold  # Check if std dev is below threshold
    return False

for episode in range(num_episodes):
    q_network.train()
    state, _ = env.reset()
    total_reward = 0
    done = False

    start_time = time.time()

    # Determine current task using the updated logic
    current_goal = get_current_task(episode + 1)

    # Assign regularization type based on the mode
    if mode == "none":
        reg_type = fixed_reg_type  # Use the pre-defined fixed regularization type throughout
    elif mode == "cyclic_task_switching":
        reg_type = get_next_reg(episode)
    elif mode == "randomized_task_switching":
        reg_type = random.choice(regularization_types)
    else:
        raise ValueError(f"Unrecognized mode: {mode}")

    # Update the regularization type in the model if needed
    q_network.reg_type = reg_type
    target_network.reg_type = reg_type

    # Reset convergence counter if switching to a new task
    if episode > 0 and current_goal != previous_goal:
        current_task_episode_count[current_goal] = 0
    previous_goal = current_goal

    # After switching tasks, track reward trends post-switch
    if episode > 0 and current_goal != previous_goal:
        task_switch_adaptation_time = episode - current_task_episode_count.get(previous_goal, 0)
        general_writer.add_scalar(f"Task_Switching/Adaptation_Time/{current_goal}", task_switch_adaptation_time,
                                  episode)
        current_task_episode_count[previous_goal] = episode
        switch_rewards = []
        for post_switch_ep in range(5):  # Monitor for the next 5 episodes post-switch
            state, _ = env.reset()
            post_reward = 0
            done = False
            while not done:
                action = select_action(state, epsilon=0, q_network=q_network)
                next_state, reward, done, _, _ = env.step(action)
                post_reward += reward
                state = next_state
            switch_rewards.append(post_reward)

        avg_switch_reward = np.mean(switch_rewards)
        general_writer.add_scalar(f"Task_Switching/{current_goal}/Avg_Post_Switch_Reward", avg_switch_reward, episode)

    # Run the episode
    while not done:
        # Add noise to state during training if evaluating RQ2
        if is_rq2:
            state = add_noise_to_state(state, noise_level=parameter_noise_stddev)

        action = q_network(torch.tensor(state, dtype=torch.float32).to(
            device)).argmax().item() if random.random() > epsilon else env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)
        reward = get_goal_reward(shape_reward(state, next_state, reward), state, current_goal)

        # Add noise to next state if evaluating RQ2
        if is_rq2:
            next_state = add_noise_to_state(next_state, noise_level=parameter_noise_stddev)

        replay_buffer.append((state, action, reward, next_state, done))
        total_steps += 1
        state = next_state
        total_reward += reward
        episode_rewards.append(reward)
        train_dqn(q_network, target_network, optimizer, reg_type)

        # Update the target network periodically
        if total_steps % target_update_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)
    rewards_log = {"episode": [], "reward": []}

    # After appending total_reward
    rewards_log["episode"].append(episode + 1)
    rewards_log["reward"].append(total_reward)

    results[reg_type]["task_rewards"][current_goal].append(total_reward)

    # Perform validation every eval_interval episodes
    if episode % eval_interval == 0 and episode > 0:
        print(f"\n--- Periodic Validation at Episode {episode} ---")
        validation_results = validate_agent(q_network, env, validation_goals, num_episodes=3)

        validation_results_history.append({
            "episode": episode,
            "validation_results": validation_results
        })

        # Log validation results to TensorBoard and append to validation_log
        for goal, avg_reward in validation_results.items():
            if avg_reward is not None:  # Only log if the result is valid
                general_writer.add_scalar(f"Validation/Average_Reward/{goal}", avg_reward, episode)
                validation_log.append({"episode": episode, "goal": goal, "average_reward": avg_reward})
            else:
                print(f"Skipping TensorBoard log for goal '{goal}' because validation failed or timed out.")

    cumulative_rewards += total_reward # # Update cumulative rewards and calculate convergence speed
    convergence_speed = cumulative_rewards - previous_cumulative_rewards # Calculate convergence speed as the change in cumulative rewards from the previous episode
    convergence_speeds.append(convergence_speed)
    writers[reg_type].add_scalar(f"{reg_type}/Convergence_Speed/{current_goal}", convergence_speed, episode) # Log the convergence speed to TensorBoard
    previous_cumulative_rewards = cumulative_rewards # Update previous cumulative rewards for the next episode

    # Log total rewards for the episode
    writers[reg_type].add_scalar(f"{reg_type}/Episode Reward", total_reward, episode)

    # Log epsilon value (exploration rate)
    writers[reg_type].add_scalar(f"{reg_type}/Epsilon", epsilon, episode)
    writers[reg_type].add_scalar(f"{reg_type}/{current_goal}/Total_Reward", total_reward, episode)

    # Log rolling average and standard deviation
    if len(episode_rewards) >= window_size:
        rolling_avg_reward = np.mean(episode_rewards[-window_size:])
        rolling_std_reward = np.std(episode_rewards[-window_size:])
        writers[reg_type].add_scalar(f"{reg_type}/Rolling_Avg_Reward", rolling_avg_reward, episode)
        writers[reg_type].add_scalar(f"{reg_type}/Rolling_Std_Reward", rolling_std_reward, episode)

        episode_count = episode
        rolling_avg_rewards.append(rolling_avg_reward)
        rolling_std_rewards.append(rolling_std_reward)

    # Log success rate if applicable
    if len(results[reg_type]["task_rewards"][current_goal]) >= window_size:
        success_rate = calculate_success_rate(results[reg_type]["task_rewards"][current_goal], performance_threshold)
        writers[reg_type].add_scalar(f"{reg_type}/{current_goal}_Success Rate", success_rate, episode)

    # Log convergence speed if task is converged
    if is_converged(results[reg_type]["task_rewards"][current_goal], performance_threshold, window_size):
        writers[reg_type].add_scalar(f"{reg_type}/Convergence Speed/{current_goal}",
                                     current_task_episode_count[current_goal], episode)

    # Track the number of episodes taken to converge
    current_task_episode_count[current_goal] += 1

    end_time = time.time()
    episode_duration = end_time - start_time

    is_converged_flag, countdown = check_convergence(results[reg_type]["task_rewards"][current_goal], performance_threshold, window_size)

    if is_converged_flag:
        if current_task_episode_count[current_goal] == 0:
            print(f"--- Convergence Starting for goal '{current_goal}' at episode {episode + 1} ---")
        else:
            print(f"Goal '{current_goal}' has been converging. Countdown: {countdown} episodes remaining.")

    # Log convergence speed and stability
    stabilization_flag = check_stabilization(episode_rewards, window_size)

    # Store per-episode details
    epoch_details.append({
        "Episode": episode + 1,
        "Task": current_goal,
        "Regularization Type": reg_type,
        "Total Reward": total_reward,
        "Epsilon": epsilon,
        "Convergence Episodes": current_task_episode_count[current_goal] if is_converged_flag else 0,
        "Countdown to Convergence": countdown if countdown is not None else "N/A",
        "Convergence Speed": convergence_speed,
        "Convergence Reached": is_converged_flag,
        "Episode Duration (s)": episode_duration,
        "Rolling Avg Reward": np.mean(episode_rewards[-window_size:]),
        "Rolling Std Reward": np.std(episode_rewards[-window_size:]),
        "Stabilized": stabilization_flag,
    })

    # Check for task convergence
    if is_converged(results[reg_type]["task_rewards"][current_goal], performance_threshold, window_size):
        if current_task_episode_count[current_goal] == 0:
            # Log the episode count when convergence is first detected
            convergence_log[current_goal].append(current_task_episode_count[current_goal])

    # Increment task episode count only after convergence
    current_task_episode_count[current_goal] += 1

    # Log convergence speed to TensorBoard
    writers[reg_type].add_scalar(f"{reg_type}/Convergence Speed/{current_goal}", convergence_speed, episode)

    # Print progress every episode
    print(f"Episode: {episode + 1}/{num_episodes}, Goal: {current_goal}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}, Regularization: {reg_type}, Duration: {episode_duration:.2f} seconds")

epoch_details_df = pd.DataFrame(epoch_details)

# When switching to a new task, evaluate on previously learned tasks
if current_goal != "original" and current_goal not in previous_goals:
    previous_goals.append(current_goal)
    retention_scores = evaluate_knowledge_retention(q_network, env, previous_goals)
    # Log retention scores to TensorBoard
    for task, score in retention_scores.items():
        writers[reg_type].add_scalar(f"{reg_type}/Retention_Score/{task}", score, episode)
    for goal, score in retention_scores.items():
        retention_results[reg_type]["previous_goal_rewards"].setdefault(goal, []).append(score)

# After training, analyze convergence speed for each task
for goal in convergence_log:
    if convergence_log[goal]:  # If there are convergence episodes logged
        avg_convergence_speed = np.mean(convergence_log[goal])
    else:
        avg_convergence_speed = None  # Indicates no convergence occurred

    # # Log average convergence speed (skip None values)
    # if avg_convergence_speed is not None:
    #     general_writer.add_scalar(f"Convergence_Speed/{goal}", avg_convergence_speed, episode)

    # Print results for clarity
    print(
        f"Goal: {goal}, "
        f"Average Convergence Speed: {avg_convergence_speed if avg_convergence_speed is not None else 'Did not converge'} episodes"
    )

# Training time tracking
training_end_time = time.time()
print(f"Total Training Time: {training_end_time - training_start_time:.2f} seconds")

# Save trained model
torch.save(q_network.state_dict(), "dqn_acrobot_model.pth")
with open("performance_log.pkl", "wb") as f:
    pickle.dump(results, f)

# --- Metrics and Plotting ---
# Improved Plotting of Average Convergence Speed per Task per Regularization Type
for reg_type in regularization_types:
    # Filter valid goals that have convergence data
    valid_goals = [goal for goal in goals + ["original"] if len(convergence_log[goal]) > 0]
    avg_convergence_speeds = [np.mean(convergence_log[goal]) if len(convergence_log[goal]) > 0 else float('inf') for
                              goal in valid_goals]

    # Only keep goals with valid convergence data (not inf)
    valid_goals = [goal for i, goal in enumerate(valid_goals) if avg_convergence_speeds[i] != float('inf')]
    avg_convergence_speeds = [speed for speed in avg_convergence_speeds if speed != float('inf')]

# Update the results dictionary to track by task first
task_results = {}

# Plotting average rewards per task per regularization type
for reg_type in regularization_types:
    for goal in goals + ["original"]:  # Assuming 'goals' is a list of tasks
        avg_reward, action_distribution = evaluate_agent(env, q_network, num_episodes=5, goal=goal)

        # Track the results by task and regularization type
        if goal not in task_results:
            task_results[goal] = {}

        if reg_type not in task_results[goal]:
            task_results[goal][reg_type] = {'average_reward': [], 'convergence_speed': [], 'action_distribution': []}

        task_results[goal][reg_type]['average_reward'].append(avg_reward)
        task_results[goal][reg_type]['action_distribution'].append(action_distribution)

        # Save results to TensorBoard
        # eval_step = 0
        # general_writer.add_scalar(f"Training/Avg_Reward/{goal}/{reg_type}", avg_reward, eval_step)
        # for i, count in enumerate(action_distribution):
        #     general_writer.add_scalar(f"Training/Action_Distribution/{goal}/{reg_type}/Action_{i}", count, eval_step)

        # eval_step += 1

# Optionally, store results using pickle
with open("task_results.pkl", "wb") as f:
    pickle.dump(task_results, f)

flattened_task_results = []

# Log structured testing results to TensorBoard for each task and reg_type
for goal, reg_dict in task_results.items():  # goal corresponds to each task
    for reg_type, metrics in reg_dict.items():  # reg_type corresponds to each regularization technique
        # Check for "average_reward" and compute scalar values if present
        if "average_reward" in metrics:
            if isinstance(metrics["average_reward"], list):
                avg_reward = np.mean(metrics["average_reward"])  # Compute the mean if it's a list
            else:
                avg_reward = metrics["average_reward"]  # Otherwise, use the scalar value directly
        else:
            avg_reward = float('nan')  # Default to NaN if missing

        # Check for "std_reward"
        if "std_reward" in metrics:
            if isinstance(metrics["std_reward"], list):
                std_reward = np.std(metrics["std_reward"])  # Compute the std if it's a list
            else:
                std_reward = metrics["std_reward"]  # Otherwise, use the scalar value directly
        else:
            std_reward = float('nan')  # Default to NaN if missing

        # Check for "average_duration"
        if "average_duration" in metrics:
            if isinstance(metrics["average_duration"], list):
                avg_duration = np.mean(metrics["average_duration"])  # Compute the mean if it's a list
            else:
                avg_duration = metrics["average_duration"]  # Otherwise, use the scalar value directly
        else:
            avg_duration = float('nan')  # Default to NaN if missing

        # Log metrics for each task and reg_type in TensorBoard
        general_writer.add_scalar(f"{goal}/{reg_type}/Avg_Reward", avg_reward)
        general_writer.add_scalar(f"{goal}/{reg_type}/Std_Reward", std_reward)
        general_writer.add_scalar(f"{goal}/{reg_type}/Avg_Duration", avg_duration)

        # Print the values for clarity
        print(f"Task: {goal}, Regularization: {reg_type} -> "
              f"Avg Reward: {avg_reward:.2f}, "
              f"Std Reward: {std_reward:.2f}, "
              f"Avg Duration: {avg_duration:.2f}")

# Convert to DataFrame
flattened_task_results_df = pd.DataFrame(flattened_task_results)

# Plot task-specific average rewards per regularization type
print("\n--- Evaluation Phase ---")
for reg_type in regularization_types:
    print(f"Evaluating model with Regularization Type: {reg_type}")

    if is_rq2:
        # Add parameter noise before evaluation if we are assessing RQ2
        add_parameter_noise(q_network, parameter_noise_stddev)

    eval_step = 0
    avg_rewards_per_goal = {}
    for goal in goals + ["original"]:
        # Evaluate agent and log action distribution
        avg_reward, action_distribution = evaluate_agent(env, q_network, num_episodes=5, goal=goal)

        # Convert action distribution (which is likely a list) to a NumPy array
        action_distribution_np = np.array(action_distribution)

        # Log average reward and action distribution
        # general_writer.add_scalar(f"Training/Avg_Reward/{goal}/{reg_type}", avg_reward, eval_step)
        # for i, count in enumerate(action_distribution_np):
        #     general_writer.add_scalar(f"Training/Action_Distribution/{goal}/{reg_type}/Action_{i}", count, eval_step)

        # Save average rewards for each goal
        avg_rewards_per_goal[goal] = avg_reward
        results[reg_type]["retention_score"].append(avg_reward)

        eval_step += 1  # Increment eval step after each goal evaluation

    print(f"Avg Reward per Goal for {reg_type}: {avg_rewards_per_goal}")

env.close()

# --- Validation Phase ---
# Validation Phase: Evaluate agent on unseen tasks or conditions
def validate_agent(agent, env, validation_goals, num_episodes=5, timeout=45):
    validation_results = {}

    for goal in validation_goals:
        start_time = time.time()  # Record the start time for the goal
        total_reward = 0
        total_steps = 0
        print(f"Start validating for goal: {goal}")

        for episode in range(num_episodes):
            # If the elapsed time exceeds the timeout, skip this goal
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                print(
                    f"Validation for goal '{goal}' took too long ({elapsed_time:.2f}s), skipping after {episode} episodes.")
                validation_results[goal] = None  # Mark the goal as skipped
                break  # Skip to the next goal

            print(f"Starting Episode {episode + 1} for goal: {goal}")

            state, _ = env.reset()  # Reset the environment and get the starting state
            done = False
            episode_reward = 0
            episode_steps = 0

            while not done:
                # If time exceeds the limit while running an episode, abort early
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout:
                    print(f"Timeout during episode {episode + 1} for goal '{goal}', skipping...")
                    validation_results[goal] = None
                    break

                # Log progress in each step
                # print(f"  Episode {episode + 1}, Step {episode_steps}, Elapsed Time: {elapsed_time:.2f}s")
                action = select_action(state, epsilon=0, q_network=agent)
                next_state, reward, done, _, _ = env.step(action)

                episode_reward += reward
                episode_steps += 1
                state = next_state

                if elapsed_time > timeout:
                    print(f"Timeout reached during step {episode_steps} of episode {episode + 1}.")
                    validation_results[goal] = None
                    break  # Break early if timeout is reached

            if elapsed_time > timeout:
                break  # If we broke early due to timeout, break the loop for this goal

            total_reward += episode_reward
            total_steps += episode_steps

        if goal not in validation_results:  # Only add result if it wasn't skipped
            avg_reward = total_reward / num_episodes
            validation_results[goal] = avg_reward
            print(f"Validation - Goal: {goal}, Avg Reward: {avg_reward}")

    return validation_results


# Run the validation phase
print("\n--- Validation Phase ---")
validation_results = validate_agent(q_network, env, validation_goals, num_episodes=5)

# Log and display validation results
for goal, avg_reward in validation_results.items():
    general_writer.add_scalar(f"Validation/Average_Reward/{goal}", avg_reward, num_episodes)
    print(f"Validation - Goal: {goal}, Avg Reward: {avg_reward}")

# Add this after the training and validation phases
print("\n--- Structured Testing Phase ---")

# --- Structured Testing Phase ---
# Define testing scenarios
testing_scenarios = [
    {
        "name": "Gradual Increase in State Noise",
        "state_noise_level": 0.1,  # Low initial state noise
        "action_noise_prob": 0.0,  # No action noise
        "reward_modifier": None,
    },
    {
        "name": "Gradual Increase in Action Noise",
        "state_noise_level": 0.0,  # No state noise
        "action_noise_prob": 0.1,  # Low initial action noise
        "reward_modifier": None,
    },
    {
        "name": "Dynamic Switching Between Noise Types",
        "state_noise_level": 0.2,  # Moderate state noise
        "action_noise_prob": 0.2,  # Moderate action noise
        "reward_modifier": None,
    },
    {
        "name": "High State Noise with Stabilization",
        "state_noise_level": 0.3,  # High state noise
        "action_noise_prob": 0.0,  # No action noise
        "reward_modifier": lambda state, reward: reward - abs(state[3]) * 0.1,  # Penalize angular velocity
    },
    {
        "name": "High Action Noise with Recovery Focus",
        "state_noise_level": 0.0,  # No state noise
        "action_noise_prob": 0.3,  # High action noise
        "reward_modifier": None,
    },
]

def structured_testing(agent, env, noise_scenarios, num_episodes=5, max_steps=500):
    testing_results = {}

    for scenario in noise_scenarios:
        print(f"\n--- Testing Scenario: {scenario['name']} ---")
        total_rewards = []
        total_durations = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            step_count = 0

            while not done and step_count < max_steps:
                # Apply state noise if defined in the scenario
                noisy_state = (
                    add_noise_to_state(state, scenario['state_noise_level'])
                    if scenario['state_noise_level'] > 0
                    else state
                )

                # Select action and apply action noise if defined in the scenario
                action = select_action(
                    noisy_state,
                    epsilon=0,
                    q_network=agent,
                    noise_level=scenario['state_noise_level'],
                    action_noise_prob=scenario['action_noise_prob'],
                )

                # Take the action in the environment
                next_state, reward, done, _, _ = env.step(action)

                # Optionally modify the reward as per scenario logic
                if 'reward_modifier' in scenario and scenario['reward_modifier'] is not None:
                    reward = scenario['reward_modifier'](state, reward)

                total_reward += reward
                state = next_state
                step_count += 1

            total_rewards.append(total_reward)
            total_durations.append(step_count)

        # Calculate and log results for the scenario
        avg_reward = np.mean(total_rewards)
        avg_duration = np.mean(total_durations)
        std_reward = np.std(total_rewards)

        testing_results[scenario['name']] = {
            "average_reward": avg_reward,
            "average_duration": avg_duration,
            "std_reward": std_reward,
        }

        print(
            f"Scenario: {scenario['name']}, Avg Reward: {avg_reward:.2f}, Std Reward: {std_reward:.2f}, Avg Duration: {avg_duration:.2f}")

    return testing_results

# Call the structured_testing function to get results for each scenario
structured_testing_results = structured_testing(q_network, env, testing_scenarios, num_episodes=5, max_steps=500)

# Log structured testing results to TensorBoard and display them
for scenario_name, results in structured_testing_results.items():
    # Check if required keys exist in results (to avoid KeyError)
    if all(key in results for key in ["average_reward", "average_duration", "std_reward"]):
        # Log each result to TensorBoard for later visualization
        general_writer.add_scalar(f"Testing/{scenario_name}/Average_Reward", results["average_reward"])
        general_writer.add_scalar(f"Testing/{scenario_name}/Average_Duration", results["average_duration"])
        general_writer.add_scalar(f"Testing/{scenario_name}/Std_Reward", results["std_reward"])

        # Print out for monitoring in the console
        print(f"{scenario_name} - Avg Reward: {results['average_reward']:.2f}, "
              f"Std Reward: {results['std_reward']:.2f}, Avg Duration: {results['average_duration']:.2f}")
    else:
        print(f"Missing keys in results for scenario: {scenario_name}. Skipping TensorBoard logging.")

# Ensure task_results is properly initialized for goal, scenario, and regularization type
for scenario_name, results in structured_testing_results.items():
    for goal in results:
        for reg_type in regularization_types:  # Loop over all regularization types
            # Initialize the task (goal) if it's not already in task_results
            if goal not in task_results:
                task_results[goal] = {}

            # Initialize the scenario for the given goal (task) if not present
            if scenario_name not in task_results[goal]:
                task_results[goal][scenario_name] = {}

            # Initialize the regularization type under the scenario if not present
            if reg_type not in task_results[goal][scenario_name]:
                task_results[goal][scenario_name][reg_type] = {}

            # Store the results for the given goal, scenario, and regularization type
            task_results[goal][scenario_name][reg_type] = {
                "average_reward": results.get("average_reward", float('nan')),
                "std_reward": results.get("std_reward", float('nan')),
                "average_duration": results.get("average_duration", float('nan'))
            }

# --- Compare Across Regularization Techniques ---
def compare_regularization_techniques(results, regularization_types):
    comparison_results = {}
    metrics = ["rewards", "convergence_speed", "retention_score", "noise_resilience"]

    for reg_type in regularization_types:
        if reg_type not in results:
            print(f"Warning: Missing data for regularization type: {reg_type}")
            continue

        comparison_results[reg_type] = {}

        for metric in metrics:
            if metric in results[reg_type]:
                data = results[reg_type][metric]
                comparison_results[reg_type][metric] = {
                    "mean": np.mean(data) if data else float('nan'),
                    "std": np.std(data) if data else float('nan'),
                    "max": np.max(data) if data else float('nan'),
                    "min": np.min(data) if data else float('nan'),
                }

    return comparison_results

# Run comparison
comparison_results = compare_regularization_techniques(results, regularization_types)

# Log results to TensorBoard
for reg_type, metrics in comparison_results.items():
    for metric, stats in metrics.items():
        general_writer.add_scalar(f"Comparison/{reg_type}/{metric}/Mean", stats["mean"])
        general_writer.add_scalar(f"Comparison/{reg_type}/{metric}/Std", stats["std"])
        general_writer.add_scalar(f"Comparison/{reg_type}/{metric}/Max", stats["max"])
        general_writer.add_scalar(f"Comparison/{reg_type}/{metric}/Min", stats["min"])

# Print comparison results for clarity
print("\n--- Comparison of Regularization Techniques ---")
for reg_type, metrics in comparison_results.items():
    print(f"Regularization Type: {reg_type}")
    for metric, stats in metrics.items():
        print(
            f"  {metric.capitalize()}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, Max={stats['max']:.2f}, Min={stats['min']:.2f}")

# --- Testing Phase ---
# Load the trained model for evaluation and testing
q_network.load_state_dict(torch.load("dqn_acrobot_model.pth", map_location=device))
q_network.to(device)

# Define the number of episodes and maximum steps for testing
test_episodes = 50
max_steps = 500  # Limit to prevent long runs

# Different noise levels for evaluating noise resilience
noise_levels = [0.0, 0.1, 0.2, 0.4]

# Timing for testing phase
testing_start_time = time.time()

# Initialize a dictionary to store results for different noise levels
testing_results = {
    "noise_level": [],
    "average_reward": [],
    "std_reward": [],
    "average_duration": []
}

# Improved Testing loop with detailed print statements for each episode
for noise_level in noise_levels:
    print(f"\nTesting with Noise Level: {noise_level}")
    eval_rewards = []
    episode_durations = []

    for episode in range(test_episodes):
        state, _ = env.reset()  # Reset environment for each test
        state = torch.tensor(state, dtype=torch.float32).to(device)  # Ensure state is on the right device
        total_reward = 0
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            episode_start_time = time.time()
            action = select_action(state.cpu().numpy(), epsilon=0, q_network=q_network, noise_level=noise_level)
            next_state, reward, done, _, _ = env.step(action)

            # Optionally add noise to the state (e.g., for noise resilience testing)
            next_state = add_noise_to_state(next_state, noise_level)
            state = torch.tensor(next_state, dtype=torch.float32).to(
                device)  # Ensure the next state is on the correct device
            total_reward += reward
            step_count += 1

        eval_rewards.append(total_reward)
        episode_duration = time.time() - episode_start_time
        episode_durations.append(episode_duration)

        print(f"Test {episode + 1}/{test_episodes}: Reward: {total_reward:.2f}, Duration: {episode_duration:.2f} s")

    avg_reward = np.mean(eval_rewards)
    avg_duration = np.mean(episode_durations)
    std_reward = np.std(eval_rewards)

    # Log noise resilience results for each task and noise level
    general_writer.add_scalar(f"Testing/Noise_Resilience/{goal}/{noise_level}/Avg_Reward", avg_reward)
    general_writer.add_scalar(f"Testing/Noise_Resilience/{goal}/{noise_level}/Std_Reward", std_reward)
    general_writer.add_scalar(f"Testing/Noise_Resilience/{goal}/{noise_level}/Avg_Duration", avg_duration)

    # Print summary for this noise level:
    print(
        f"Noise Level: {noise_level} - Avg Reward: {avg_reward:.2f}, Std Reward: {std_reward:.2f}, Avg Duration: {avg_duration:.2f}")

    # Append to results
    testing_results["noise_level"].append(noise_level)
    testing_results["average_reward"].append(avg_reward)
    testing_results["std_reward"].append(std_reward)
    testing_results["average_duration"].append(avg_duration)

    # Print summary for current noise level
    print(f"\nSummary for Noise Level {noise_level}:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Standard Deviation of Reward: {std_reward:.2f}")
    print(f"Average Episode Duration: {avg_duration:.2f} seconds")

# Calculate total testing time
testing_end_time = time.time()
testing_duration = testing_end_time - testing_start_time
print(f"\nTotal Testing Time: {testing_duration:.2f} seconds")

# Print the testing results for each noise level in the console
print("\n--- Testing Results Summary ---")
for i, noise_level in enumerate(testing_results["noise_level"]):
    print(f"Noise Level: {noise_level}")
    print(f"  Average Reward: {testing_results['average_reward'][i]:.2f}")
    print(f"  Standard Deviation of Reward: {testing_results['std_reward'][i]:.2f}")
    print(f"  Average Episode Duration: {testing_results['average_duration'][i]:.2f} seconds")

# Summary statistics for interpretation
print("\n--- Summary Statistics for Testing Phase ---")
print("Noise Levels Tested:", testing_results["noise_level"])
print("Average Rewards:", testing_results["average_reward"])
print("Standard Deviations of Rewards:", testing_results["std_reward"])
print("Average Episode Durations:", testing_results["average_duration"])

# --- Save all excel files ---
# Consolidate all results into a single dictionary
consolidated_results = {
    "epoch_details": epoch_details,  # From the training loop
    "validation_results": validation_results,  # From validation phase
    "convergence_results": [],  # Convergence results
    "episode_duration": episode_duration,
    "structured_testing_results": structured_testing_results,  # Structured testing phase results
    "comparison_results": comparison_results,  # Regularization comparison
    "testing_results": testing_results,  # Noise resilience testing results
    "action_distributions": action_distributions,  # Action distributions
    "task_results": flattened_task_results_df,  # Storing the flattened task results DataFrame
    "episode": range(window_size, episode_count + 1),
    "rolling_avg_reward": rolling_avg_rewards,
    "rolling_std_reward": rolling_std_rewards,
}

all_results = []

# Flatten the 'epoch_details' if it's a list of dictionaries
if 'epoch_details' in consolidated_results:
    epoch_df = pd.DataFrame(consolidated_results['epoch_details'])
    epoch_df['Source'] = 'Epoch Details'
    all_results.append(epoch_df)

# Flatten the 'validation_results_history' for every 50 episodes
if validation_results_history:
    validation_results_flat = []

    for entry in validation_results_history:
        # For each episode in the history, flatten the 'validation_results' dictionary and include episode
        flat_entry = {'episode': entry['episode']}
        flat_entry.update(entry['validation_results'])  # Add the goal-reward pairs as columns
        validation_results_flat.append(flat_entry)

    validation_df = pd.DataFrame(validation_results_flat)
    validation_df['Source'] = 'Validation Results (every 50 episodes)'
    all_results.append(validation_df)

# Flatten the 'validation_results' if it's a dictionary (single snapshot of validation results)
if 'validation_results' in consolidated_results:
    # Check if 'validation_results' is a dictionary and print its structure
    validation_results_flat = []

    print("---- Debugging Validation Results ----")
    print(f"Validation Results: {consolidated_results['validation_results']}")
    print("-------------------------------------------------")

    if isinstance(consolidated_results['validation_results'], dict):
        # Each entry in 'validation_results' is a goal-reward pair
        for goal, reward in consolidated_results['validation_results'].items():
            # Create a flat entry for each validation goal and its reward
            flat_entry = {'goal': goal, 'average_reward': reward}
            validation_results_flat.append(flat_entry)

        validation_df = pd.DataFrame(validation_results_flat)
        validation_df['Source'] = 'Validation Results (Single Snapshot)'
        all_results.append(validation_df)
    else:
        print("Validation results are not in the expected format. Skipping this step.")

# Normalize convergence results to ensure they have consistent keys across all tasks
def normalize_convergence_results(results):
    all_keys = set()

    # Collect all unique keys from the dictionary
    for result in results:
        if isinstance(result, dict):
            all_keys.update(result.keys())
        else:
            print(f"Warning: Skipping non-dictionary item: {result}")

    # Normalize all dictionaries in the list to have the same keys
    normalized_results = []
    for result in results:
        if isinstance(result, dict):
            normalized_result = result.copy()
            for key in all_keys:
                if key not in normalized_result:
                    normalized_result[key] = None  # Use None or another placeholder
            normalized_results.append(normalized_result)
        else:
            normalized_results.append(None)  # Handle non-dictionary results here

    return normalized_results

# Flatten convergence results
def flatten_convergence_results(convergence_log):
    flattened_results = []
    for goal, details in convergence_log.items():
        for entry in details:  # Assuming each entry in 'details' represents an episode's convergence info
            flattened_entry = {
                'goal': goal,
                'episode': entry.get('episode', None),  # Ensure that each entry has an 'episode' field
                'reward': entry.get('reward', None),
                'converged': entry.get('converged', None),
                'convergence_episode': entry.get('convergence_episode', None)
            }
            flattened_results.append(flattened_entry)
    return flattened_results

# Normalize and flatten convergence results before adding to consolidated_results
if 'convergence_results' in consolidated_results and len(consolidated_results['convergence_results']) > 0:
    print("Normalizing Convergence Results...")
    normalized_convergence = normalize_convergence_results(consolidated_results['convergence_results'])
    flattened_convergence = flatten_convergence_results(normalized_convergence)
    consolidated_results['convergence_results'] = flattened_convergence
else:
    print("No convergence results found, skipping normalization and flattening.")


# Flatten the 'structured_testing_results'
if 'structured_testing_results' in consolidated_results:
    structured_testing_df = pd.DataFrame(consolidated_results['structured_testing_results'])
    structured_testing_df['Source'] = 'Structured Testing Results'
    all_results.append(structured_testing_df)

# Flatten the 'testing_results' if it's a dictionary/list
if 'testing_results' in consolidated_results:
    testing_df = pd.DataFrame(consolidated_results['testing_results'])
    testing_df['Source'] = 'Testing Results'
    all_results.append(testing_df)

# Flatten 'task_results' and convert into DataFrame (considering it as a nested dictionary)
if 'task_results' in consolidated_results:
    task_results_df = pd.DataFrame(consolidated_results['task_results'])
    task_results_df['Source'] = 'Task Results'
    all_results.append(task_results_df)

# Flatten other results (comparison, action_distributions, etc.) similarly
if 'comparison_results' in consolidated_results:
    comparison_df = pd.DataFrame(consolidated_results['comparison_results'])
    comparison_df['Source'] = 'Comparison Results'
    all_results.append(comparison_df)

if 'action_distributions' in consolidated_results:
    action_distributions_df = pd.DataFrame(consolidated_results['action_distributions'])
    action_distributions_df['Source'] = 'Action Distributions'
    all_results.append(action_distributions_df)

final_df = pd.concat(all_results, ignore_index=True)
filename = f"results_{'rq1' if is_rq1 else 'rq2'}_{mode}.xlsx"
final_df.to_excel(filename, index=False)
print(f"All results saved to '{filename}'")

# Close the environment after testing
env.close()