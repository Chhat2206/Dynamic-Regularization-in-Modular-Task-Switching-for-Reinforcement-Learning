#!pip install gymnasium
import collections
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
import datetime
from multiprocessing import Process, Manager

env = gym.make("CartPole-v1")
use_custom_goals = True  # Set False to train all episodes under the original goal in all episodes
is_rq1 = True  # Set this to True to evaluate RQ1, and automatically set RQ2 to False
is_rq2 = not is_rq1
mode = "none"  # Options are "none", "cyclic_task_switching", "randomized_task_switching", "structured_task_switching"
goals = ["quick_recovery", "periodic_swing", "maintain_balance"]
fixed_reg_type = "dropout"  # Choose from ["dropout", "l1", "l2", "batch_norm"]

# Hyperparameters
learning_rate = 0.0001
gamma = 0.99
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
min_epsilon = 0.01
batch_size = 128
replay_buffer_size = 1000000
replay_buffer = deque(maxlen=replay_buffer_size)
target_update_frequency = 500
stability_threshold = 0.01
window_size = 10
reward_window = collections.deque(maxlen=window_size)
eval_interval = 100
num_episodes = eval_interval * 8
parameter_noise_stddev = 0.1  # Standard deviation for parameter noise (for RQ2)

training_start_time = time.time()

def shape_reward(state, next_state, reward):
    # Extract relevant state information
    cart_position = next_state[0]
    cart_velocity = next_state[1]
    pole_angle = next_state[2]
    pole_velocity = next_state[3]

    # Penalize large cart position (keep the cart near the center)
    cart_position_penalty = np.abs(cart_position) * 0.1  # Higher penalty for moving further from the center

    # Penalize large pole angle (keep the pole upright)
    pole_angle_penalty = np.abs(pole_angle) * 0.1  # Larger angle means higher penalty

    # Penalize high velocities (keep the movements small)
    cart_velocity_penalty = np.abs(cart_velocity) * 0.01  # Higher velocity means larger penalty
    pole_velocity_penalty = np.abs(pole_velocity) * 0.01  # Larger pole velocity increases penalty

    # Combine the penalties with the original reward (reward encourages balance)
    shaped_reward = reward - (
                cart_position_penalty + pole_angle_penalty + cart_velocity_penalty + pole_velocity_penalty)

    return shaped_reward

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
        angle_deviation = abs(state[2])  # Pole angle deviation
        speed_of_recovery = abs(state[3])  # Angular velocity (speed of recovery)

        # Reward based on how quickly the pole is returning to the upright position
        if angle_deviation > 0.1:
            reward += 5 / (angle_deviation + 0.1)
            if speed_of_recovery < 0.5:
                reward -= 0.5  # Penalty for slow recovery
        else:
            reward += 0.1  # Small reward for maintaining near upright position

        # Apply penalty for extreme deviations
        if abs(state[0]) > 2.4 or angle_deviation > 0.2:
            reward -= 1

    elif goal == "periodic_swing":
        swing_angle = abs(state[2])  # Swing angle deviation from periodic motion
        swing_speed = abs(state[3])  # Swing speed

        # Reward for keeping the pole in periodic motion within limits
        if 0.05 < swing_angle < 0.1:
            reward += 1
            if swing_speed < 0.5:
                reward -= 0.2  # Penalty for slow swings
        else:
            reward -= 0.05  # Slight penalty for not maintaining periodic swing

    elif goal == "maintain_balance":
        cart_position = abs(state[0])  # Position of the cart
        cart_velocity = abs(state[1])  # Velocity of the cart

        # Reward for small, controlled cart movements
        if cart_position < 0.5 and cart_velocity < 0.1:
            reward += 1
        elif cart_position < 1.0 and cart_velocity < 0.2:
            reward += 0.5
        else:
            reward -= 0.1  # Penalty for large or fast movements

    # Log reward and state for debugging
    # print(f"Goal: {goal}, Reward: {reward}, State: {state}")

    return reward

# Function to add noise to state
def add_noise_to_state(state, noise_level=0.1):
    noise = np.random.normal(0, noise_level, size=state.shape)
    noisy_state = state + noise
    return np.clip(noisy_state, env.observation_space.low, env.observation_space.high)

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
        "long_term_adaptability": {}
    } for reg_type in regularization_types
}

# --- Training DQN ---
# Define device (use GPU if available, otherwise fallback to CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training with device: {device}")

# Evaluate agent on the given goal/environment
def evaluate_agent(env, q_network, num_episodes=5, goal="original"):
    total_rewards = []

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
                state = next_state
                step_count += 1

        total_rewards.append(total_reward)
        print(f"Evaluation Episode: {episode + 1}, Goal: {goal}, Total Reward: {total_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"Average Reward after Evaluation for Goal '{goal}': {avg_reward}")

    return avg_reward

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
    if model is None:
        raise ValueError("Model is None. Cannot add noise.")
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                noise = torch.randn_like(param) * stddev
                param.data.add_(noise)
    return model

# Dictionary to track convergence speed for each task
convergence_log = {goal: [] for goal in goals + ["original"]}

# Number of episodes taken to converge after switching to a task
current_task_episode_count = {goal: 0 for goal in goals + ["original"]}

# Function to evaluate knowledge retention
def evaluate_knowledge_retention(agent, env, learned_tasks, num_episodes=5):
    retention_rewards = {}
    # Log retention trends over tasks
    retention_trends = []

    for task in learned_tasks:
        # Directly assign the average reward from evaluate_agent
        avg_reward = evaluate_agent(env, q_network, num_episodes=5, goal=task)
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
if mode == "none": reg_type = fixed_reg_type
else: reg_type = None  # Placeholder to be set dynamically based on task-switching

# Initialize Q-network and target network with the selected regularization type
q_network = DQN(env.observation_space.shape[0], env.action_space.n, reg_type=reg_type).to(device)
print(f"q_network after initialization: {q_network}")
target_network = DQN(env.observation_space.shape[0], env.action_space.n, reg_type=reg_type).to(device)
print(f"target_network initialized: {target_network}")

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

hyperparams = {
    "learning_rate": learning_rate,
    "gamma": gamma,
    "epsilon_decay": epsilon_decay,
    "batch_size": batch_size,
    "num_episodes": num_episodes,
}

# Function to train the DQN using replay buffer
def train_dqn(q_network, target_network, optimizer, reg_type, is_rq2=False, parameter_noise_stddev=0.1):
    if len(replay_buffer) < batch_size:
        # Insufficient samples in replay buffer
        return

    if q_network is None:
        raise ValueError("q_network has not been initialized.")

    # Apply parameter noise if is_rq2 is True
    if is_rq2:
        q_network = add_parameter_noise(q_network, stddev=parameter_noise_stddev)
        # print("RQ2 is active")

    # Sample random mini-batch from the replay buffer
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

    # Compute the target Q-values using the target network
    with torch.no_grad():
        next_q_values = target_network(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * gamma * next_q_values

    # Compute loss using Mean Squared Error (MSE)
    loss = nn.MSELoss()(current_q_values, target_q_values)

    # Add L1 regularization if specified
    if reg_type == "l1":
        l1_loss = sum(torch.abs(param).sum() for param in q_network.parameters())
        loss += 1e-4 * l1_loss

    # Backpropagation and optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

validation_goals = [
    "stabilize_at_upright_position",  # Reward for keeping the pole close to upright position
    "longer_balances",               # Reward for balancing the pole for longer periods
    "small_cart_movements",          # Reward for keeping the cart movements small
    "low_state_noise_resilience",    # Test performance under low state noise
    "high_action_noise_resilience",  # Test performance under high action noise
]

def validate_agent(agent, env, validation_goals, num_episodes=5, timeout=30):
    validation_results = {}

    for goal in validation_goals:
        print(f"\nValidating for goal: {goal}")

        # Record start time for this goal validation
        start_time = time.time()

        # Set validation parameters based on the goal
        if goal == "stabilize_at_upright_position":
            def reward_shaping(state, reward):
                # Reward for keeping the pole upright (state[0] should be close to zero)
                if abs(state[0]) < 0.05:  # Close to vertical position
                    reward += 1.0
                return reward

            noise_level = 0.0
            action_noise_prob = 0.0

        elif goal == "longer_balances":
            def reward_shaping(state, reward):
                # Reward for staying alive longer, encouraging long balances
                if state[2] > 0.5:  # Pole angle closer to upright
                    reward += 0.5
                return reward

            noise_level = 0.0
            action_noise_prob = 0.0

        elif goal == "small_cart_movements":
            def reward_shaping(state, reward):
                # Penalize large cart movements, reward small movements
                if abs(state[0]) < 0.1:  # Cart near the center
                    reward += 1.0
                return reward

            noise_level = 0.0
            action_noise_prob = 0.0

        elif goal == "low_state_noise_resilience":
            reward_shaping = None
            noise_level = 0.05  # Low state noise
            action_noise_prob = 0.0

        elif goal == "high_action_noise_resilience":
            reward_shaping = None
            noise_level = 0.0  # No state noise
            action_noise_prob = 0.2  # High action noise

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
                    print(f"Validation for goal '{goal}' took too long ({elapsed_time:.2f}s), skipping this goal.")
                    validation_results[goal] = None  # Mark the goal as skipped
                    break  # Exit the loop for the current goal if timeout is reached

                # Add noise to state if applicable
                noisy_state = add_noise_to_state(state, noise_level) if noise_level > 0 else state
                action = select_action(noisy_state, epsilon=0, q_network=agent, noise_level=noise_level, action_noise_prob=action_noise_prob)
                next_state, reward, done, _, _ = env.step(action)

                # Apply reward shaping if defined
                if reward_shaping is not None:
                    reward = reward_shaping(state, reward)
                total_reward += reward
                state = next_state

            # Append the total reward of this episode to the list for this goal
            if elapsed_time <= timeout:
                total_rewards.append(total_reward)

        # Store the validation result if the timeout was not exceeded
        if elapsed_time <= timeout:
            avg_reward = np.mean(total_rewards) if total_rewards else 0
            validation_results[goal] = avg_reward
            print(f"Validation - Goal: {goal}, Avg Reward: {avg_reward}")
        else:
            print(f"Validation for goal '{goal}' skipped due to timeout.")

    return validation_results


def reward_shaping_for_goal(state, reward, goal):
    if goal == "stabilize_at_angle":
        # Reward for being near horizontal position
        if 0.8 < state[0] < 1.2:  # Cart's position near the middle
            reward += 0.5

    elif goal == "noisy_stabilization":
        # Reward for being close to balance
        if abs(state[0]) < 0.5:  # Cart near center
            reward += 1.0

    elif goal == "noisy_swing_maximization":
        # Reward for high swing amplitude
        if abs(state[1]) > 1.5:  # Pole's angle large
            reward += 1.0

    return reward

def add_noise_to_state(state, noise_level):
    """
    Adds noise to the state (simulating noisy sensor readings).
    """
    noisy_state = state + np.random.normal(0, noise_level, size=state.shape)
    return np.clip(noisy_state, -1, 1)  # Clip to valid state range


def select_action_with_noise(state, agent, noise_level=0.0, action_noise_prob=0.0):
    """
    Selects an action with optional noise added to the state and/or action.
    """
    # Add noise to state if applicable
    noisy_state = add_noise_to_state(state, noise_level) if noise_level > 0 else state

    # Select action from the agent's policy (with optional action noise)
    action = select_action(noisy_state, epsilon=0, q_network=agent)  # Assuming `select_action` is already defined

    # Add action noise if necessary
    if np.random.random() < action_noise_prob:
        action = np.random.choice([0, 1])  # Assuming binary action space, change if necessary

    return action, noisy_state

def reward_shaping_for_goal(state, reward, goal):
    """
    Reward shaping function based on the validation goal.
    """
    if goal == "stabilize_at_angle":
        # Reward for being near horizontal position
        if 0.8 < state[0] < 1.2:  # Cart's position near the middle
            reward += 0.5

    elif goal == "noisy_stabilization":
        # Reward for being close to balance
        if abs(state[0]) < 0.5:  # Cart near center
            reward += 1.0

    elif goal == "noisy_swing_maximization":
        # Reward for high swing amplitude
        if abs(state[1]) > 1.5:  # Pole's angle large
            reward += 1.0

    return reward

assigned_regularizations = set()
task_to_reg = {}

tasks = [
    "original",        # Episodes 1-x
    "quick_recovery",
    "periodic_swing",
    "maintain_balance",
    "quick_recovery",
    "periodic_swing",
    "maintain_balance",
    "original",
]

# Function to determine current task based on episode number
def get_current_task(episode_number):
    task_index = (episode_number - 1) // eval_interval  # Divide to get the task cycle index
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

# Initialize task-to-regularization assignments dictionary
task_to_reg = {}

# Chooses a regularization type for the current task in structured task switching mode.
def get_task_regularization_structured(episode):
    # Determine the current task (goal) for the episode
    current_goal = get_current_task(episode + 1)

    # If this task hasn't been assigned a regularization type, assign one randomly
    if current_goal not in task_to_reg:
        # Randomly shuffle the list of available regularizations and pick one that isn't already assigned
        available_regs = [reg for reg in regularization_types if reg not in task_to_reg.values()]

        if available_regs:  # Ensure there are still regularization types available to assign
            reg_type = random.choice(available_regs)
            task_to_reg[current_goal] = reg_type
            print(f"[DEBUG] Assigned {reg_type} to task '{current_goal}' (Episode {episode + 1})")
        else:
            raise ValueError("No available regularizations left for task assignments.")
    else:
        # If the regularization type is already assigned, just print it
        reg_type = task_to_reg[current_goal]
        # print(f"[DEBUG] Task '{current_goal}' already assigned '{reg_type}' (Episode {episode + 1})")

    # Return the regularization type assigned to this task
    return reg_type

validation_results_history = []
cumulative_rewards = 0
previous_cumulative_rewards = 0
convergence_speeds = []

performance_threshold = -100

convergence_goals = {
    "original": -85,
    "maintain_balance": -50,
    "periodic_swing": -75,
    "quick_recovery": -70
}

def update_convergence_goal(task_rewards, task_goal):
    # Sort the rewards in ascending order
    temp_rewards = list(reward_window)
    sorted_rewards = sorted(temp_rewards)

    # Log the sorted rewards list for debugging
    print(f"\n[DEBUG] Sorted Rewards for Task '{task_goal}': {sorted_rewards}")

    # Calculate and log the median of the sorted rewards
    print(f"[DEBUG] Calculating Median for Task '{task_goal}':")
    print(f"  Sorted Rewards: {sorted_rewards}")

    # Check if the list has an odd or even number of elements
    if len(sorted_rewards) % 2 == 1:
        # Odd number of elements, median is the middle element
        median_value = sorted_rewards[len(sorted_rewards) // 2]
        print(f"  [DEBUG] Odd number of elements, Median is the middle element: {median_value}")
    else:
        # Even number of elements, median is the average of the two middle elements
        middle1 = sorted_rewards[len(sorted_rewards) // 2 - 1]
        middle2 = sorted_rewards[len(sorted_rewards) // 2]
        median_value = (middle1 + middle2) / 2
        print(f"  [DEBUG] Even number of elements, Median is the average of the two middle elements:")
        print(f"    ( {middle1} + {middle2} ) / 2 = {median_value}")

    # Log the calculated median value
    print(f"[DEBUG] Calculated Median Value: {median_value:.2f}")

    # Update the convergence goal to the median value
    print(f"[DEBUG] Updating Convergence Goal for Task '{task_goal}' to {median_value:.2f}")
    convergence_goals[task_goal] = median_value

    # Print the updated convergence goal
    print(f"[DEBUG] Convergence goal for Task '{task_goal}' updated to: {median_value:.2f}")

def check_convergence(task_rewards, window_size, current_goal, reward_window, percentage=5.0):

    # Calculate goal_min and goal_max over the recent rewards
    goal_min = min(reward_window)
    goal_max = max(reward_window)

    # Calculate average and standard deviation of the rewards in the window
    avg_reward = np.mean(reward_window)  # Use the reward_window for threshold calculation
    std_reward = np.std(reward_window)

    # Calculate the total reward threshold dynamically as a percentage of avg_reward
    total_reward_threshold = avg_reward * (percentage / 100)

    # Check if the convergence criteria are met based on goal range (goal_max - goal_min)
    if goal_max - goal_min < total_reward_threshold:
        is_converged_flag = True
    else:
        is_converged_flag = False

    return is_converged_flag, avg_reward, std_reward, goal_min, goal_max

# Function to calculate the performance of a given goal using the reward calculated by get_goal_reward function
def calculate_goal_performance(goal_name, env, q_network, epsilon=0, num_episodes=5, max_episode_time=30,
                               min_action_time=0.01, step_time_limit=1.0):
    total_rewards = []  # List to track rewards across episodes
    episode_times = []  # List to track time per episode
    total_goals_rewards = []  # List to track rewards for the goal across episodes

    # Manager to track shared timeout flag
    with Manager() as manager:
        timeout_flag = manager.Value('i', 0)  # Shared flag to indicate timeout

        def check_timeout(start_time):
            """Monitor elapsed time for timeouts."""
            elapsed_time = time.time() - start_time
            if elapsed_time > max_episode_time:
                timeout_flag.value = 1  # Set flag if timeout exceeds max episode time

        for episode in range(num_episodes):
            try:
                print(f"Running episode {episode + 1} | Goal: {goal_name}")
                start_time = time.time()  # Start time for this episode

                episode_rewards = []  # Track rewards within a single episode
                state, _ = env.reset()  # Reset the environment to get the initial state
                done = False

                # Start a process to monitor timeout
                timeout_process = Process(target=check_timeout, args=(start_time,))
                timeout_process.start()

                while not done:
                    # Monitor action selection time
                    start_action_time = time.time()
                    try:
                        # Epsilon-greedy policy for action selection
                        action = q_network(torch.tensor(state, dtype=torch.float32).to(device)).argmax().item() \
                            if random.random() > epsilon else env.action_space.sample()
                        action_time = time.time() - start_action_time

                        # Only print action time if it exceeds the threshold
                        if action_time > min_action_time:
                            action_time_ms = action_time * 1000  # Convert seconds to milliseconds
                            print(f"Action time: {action_time_ms:.4f} ms ({action_time_ms / 1000:.4f} seconds)")

                        # Monitor environment step time
                        start_step_time = time.time()
                        next_state, reward, done, _, _ = env.step(action)
                        step_time = time.time() - start_step_time

                        # Check if the environment step took too long and handle timeout
                        if step_time > step_time_limit:
                            print(f"[WARNING] Environment step took too long ({step_time:.2f} seconds). Terminating episode.")
                            done = True  # Force termination if step time exceeds the limit
                            return None, None  # Return early if there's a timeout

                        # Calculate the goal-specific reward using the `get_goal_reward` function
                        reward = get_goal_reward(reward, state, goal_name)

                        episode_rewards.append(reward)  # Collect reward for this time step
                        state = next_state

                        # Check for timeout based on the flag
                        if timeout_flag.value == 1:
                            print(f"[WARNING] Episode {episode + 1} exceeded max time. Terminating.")
                            done = True  # Force termination if timeout occurs
                            break

                    except Exception as e:
                        print(f"[ERROR] Error during episode {episode + 1} action step: {e}")
                        return None, None  # Exit if action step fails

                # Stop timeout process
                timeout_process.terminate()

                # Calculate the time taken for the episode
                episode_time = time.time() - start_time
                episode_times.append(episode_time)

                # Add the total reward of this episode to the list of rewards
                total_rewards.append(np.sum(episode_rewards))
                total_goals_rewards.append(np.sum(episode_rewards))

            except Exception as e:
                print(f"[ERROR] Error during episode {episode + 1} for goal {goal_name}: {e}")
                return None, None  # Exit if any other error occurs

        # Calculate the average and variance of rewards
        try:
            avg_reward = np.mean(total_rewards) if total_rewards else None
            reward_variance = np.var(total_rewards) if total_rewards else None
            avg_episode_time = np.mean(episode_times) if episode_times else None
        except Exception as e:
            print(f"[ERROR] Error during reward calculations: {e}")
            return None, None  # Exit if reward calculations fail

        # Print the total rewards and average reward calculation
        total_sum = sum(total_goals_rewards)
        print("\n--- Goal Rewards Calculation ---")
        print(f"Total Rewards across all episodes: {total_sum} (sum of: {', '.join(map(str, total_goals_rewards))})")
        print(f"Average Reward: {total_sum / num_episodes:.2f} = {total_sum} / {num_episodes}")

        # Check if avg_episode_time is None before trying to print it
        if avg_episode_time is not None:
            print(f"Average Time per Episode: {avg_episode_time:.2f} seconds")
        else:
            print("Average Time per Episode: N/A (all episodes were skipped due to time limits)")

        # Evaluate if the trials are taking too long and suggest lowering the number of episodes
        if avg_episode_time and avg_episode_time * num_episodes > 60:  # If total time exceeds 1 minute, suggest reducing the trials
            print(
                f"\nWarning: Total time ({avg_episode_time * num_episodes:.2f} seconds) exceeds 1 minute. Consider reducing the number of episodes.")

        return avg_reward, reward_variance

episode_count = 0
goal_performance_history = []

# --- Training Loop ---
print("Training Loop")
timeout_duration = 60

for episode in range(num_episodes):
    q_network.train()
    state, _ = env.reset()
    total_reward = 0
    done = False

    start_time = time.time()

    # Determine current task using the updated logic
    current_goal = get_current_task(episode + 1)

    # Assign regularization type based on the mode
    if mode == "none": reg_type = fixed_reg_type  # Use the pre-defined fixed regularization type throughout
    elif mode == "cyclic_task_switching": reg_type = get_next_reg(episode)
    elif mode == "randomized_task_switching": reg_type = random.choice(regularization_types)
    elif mode == "structured_task_switching": reg_type = get_task_regularization_structured(episode)
    else: raise ValueError(f"Unrecognized mode: {mode}")

    # Update the regularization type in the model if needed
    q_network.reg_type = reg_type
    target_network.reg_type = reg_type

    # Reset convergence counter if switching to a new task
    if episode > 0 and current_goal != previous_goal: current_task_episode_count[current_goal] = 0
    previous_goal = current_goal

    # After switching tasks, track reward trends post-switch
    if episode > 0 and current_goal != previous_goal:
        task_switch_adaptation_time = episode - current_task_episode_count.get(previous_goal, 0)

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

    # Run the episode
    while not done:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_duration:
            print(f"Episode {episode + 1} timed out after {elapsed_time:.2f} seconds.")
            break

        if is_rq2: q_network = add_parameter_noise(q_network, stddev=parameter_noise_stddev)

        action = q_network(torch.tensor(state, dtype=torch.float32).to(
            device)).argmax().item() if random.random() > epsilon else env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)
        reward = get_goal_reward(shape_reward(state, next_state, reward), state, current_goal)

        replay_buffer.append((state, action, reward, next_state, done))
        total_steps += 1
        state = next_state
        total_reward += reward
        episode_rewards.append(reward)
        train_dqn(q_network, target_network, optimizer, reg_type, is_rq2=is_rq2, parameter_noise_stddev=parameter_noise_stddev)

        # Update the target network periodically
        if total_steps % target_update_frequency == 0: target_network.load_state_dict(q_network.state_dict())

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)
    rewards_log = {"episode": [], "reward": []}

    episode_count += 1
    if episode_count % eval_interval == 0:
        print(f"Clearing reward_window after episode {episode_count}")
        reward_window = collections.deque(maxlen=window_size)  # Reinitialize the deque

    # After appending total_reward
    rewards_log["episode"].append(episode + 1)
    rewards_log["reward"].append(total_reward)
    reward_window.append(total_reward) # Append the total reward of the episode into the reward window for tracking
    results[reg_type]["task_rewards"][current_goal].append(total_reward)

    if episode % eval_interval == 0 and episode > 0:
        print(f"\n--- Periodic Validation at Episode {episode} ---")
        validation_results = validate_agent(q_network, env, validation_goals, num_episodes=3)

        validation_results_history.append({
            "episode": episode,
            "validation_results": validation_results
        })

    # Cumulative rewards and convergence speed tracking
    cumulative_rewards += total_reward  # Update cumulative rewards
    convergence_speed = cumulative_rewards - previous_cumulative_rewards  # Calculate convergence speed
    convergence_speeds.append(convergence_speed)  # Append speed to log

    previous_cumulative_rewards = cumulative_rewards  # Update previous cumulative rewards

    # Track the number of episodes taken to converge
    current_task_episode_count[current_goal] += 1

    end_time = time.time()
    episode_duration = end_time - start_time

    # Calculate rolling reward window and stats
    current_rewards_window = results[reg_type]["task_rewards"][current_goal][-window_size:]
    avg_reward = np.mean(current_rewards_window)
    std_reward = np.std(current_rewards_window)

    task_rewards = list(reward_window)

    # Now call the convergence check
    is_converged_flag, avg_reward_check, std_reward_check, goal_min, goal_max = check_convergence(
        task_rewards, window_size, current_goal, reward_window, percentage=5.0
    )

    # Check if values are None and provide a default if so
    avg_reward_check_str = f"{avg_reward_check:.2f}" if avg_reward_check is not None else "N/A"
    std_reward_check_str = f"{std_reward_check:.2f}" if std_reward_check is not None else "N/A"
    goal_min_str = f"{goal_min:.2f}" if goal_min is not None else "N/A"
    goal_max_str = f"{goal_max:.2f}" if goal_max is not None else "N/A"

    # If converged, update the goal
    if is_converged_flag:
        print(f"[DEBUG] Convergence condition met for task '{current_goal}'. Updating convergence goal.")
        update_convergence_goal(results[reg_type]["task_rewards"][current_goal], current_goal)

    # Add a check to ensure that the values are not None before formatting them
    if avg_reward_check is not None and std_reward_check is not None and goal_min is not None and goal_max is not None:
        print(f"[DEBUG] Convergence Check | Goal: {current_goal}, "
              f"Avg Reward (Last {window_size} Episodes): {avg_reward_check:.2f}, "
              f"Std Reward (Last {window_size} Episodes): {std_reward_check:.2f}, "
              f"Goal Range: ({goal_min:.2f}, {goal_max:.2f}), "
              f"Convergence Flag: {is_converged_flag}")

        print(f"   Current Episode: {episode + 1}, Total Episodes for Goal: {current_task_episode_count[current_goal]}, "
            f"Reward Window (Last {window_size} Episodes): {current_rewards_window}")

        # Indicating if convergence has been met or not
        if is_converged_flag:
            print(f"[INFO] Convergence condition met for goal '{current_goal}'!")

    # Store per-episode details
    epoch_details.append({
        "Episode": episode + 1,
        "Task": current_goal,
        "Regularization Type": reg_type,
        "Total Reward": total_reward,
        "Epsilon": epsilon,
        "Stability Reached": is_converged_flag,
        "Minimum Goal": goal_min,
        "Stability Avg Reward (Last 10 Episodes)": avg_reward,
        "Stability Std Reward (Last 10 Episodes)": std_reward,
        "Episode Duration (s)": episode_duration,
    })

    if episode % eval_interval == 0 and episode > 0:
        print(
            f"\nStarting goal performance tracking for episode {episode}...")  # Print message indicating the start of tracking

        # Track and print the evaluation of each goal
        print("Evaluating 'quick_recovery' goal...")
        quick_recovery_avg_reward, quick_recovery_variance = calculate_goal_performance("quick_recovery", env,
                                                                                        q_network)
        print(f"Quick Recovery - Avg Reward: {quick_recovery_avg_reward}, Variance: {quick_recovery_variance}")

        print("Evaluating 'periodic_swing' goal...")
        periodic_swing_avg_reward, periodic_swing_variance = calculate_goal_performance("periodic_swing", env,
                                                                                        q_network)
        print(f"Periodic Swing - Avg Reward: {periodic_swing_avg_reward}, Variance: {periodic_swing_variance}")

        print("Evaluating 'maintain_balance' goal...")
        maintain_balance_avg_reward, maintain_balance_variance = calculate_goal_performance("maintain_balance", env,
                                                                                            q_network)
        print(f"Maintain Balance - Avg Reward: {maintain_balance_avg_reward}, Variance: {maintain_balance_variance}")

        # Calculate performance for each goal
        goal_performance = {
            "episode": episode,
            "quick_recovery_avg_reward": quick_recovery_avg_reward,
            "quick_recovery_variance": quick_recovery_variance,
            "periodic_swing_avg_reward": periodic_swing_avg_reward,
            "periodic_swing_variance": periodic_swing_variance,
            "maintain_balance_avg_reward": maintain_balance_avg_reward,
            "maintain_balance_variance": maintain_balance_variance,
        }

        goal_performance_history.append(goal_performance)

    if is_converged_flag:
        if current_task_episode_count[current_goal] == 1:
            # Log the first episode where convergence is detected
            print(f"[DEBUG] Task {current_goal} converged at episode {episode + 1}")
            convergence_log[current_goal].append(episode + 1)  # Log the first episode of convergence
        current_task_episode_count[current_goal] += 1

    # Increment task episode count only after convergence
    current_task_episode_count[current_goal] += 1

    # Print progress every episode
    print(f"Episode: {episode + 1}/{num_episodes}, Goal: {current_goal}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}, Regularization: {reg_type}, Duration: {episode_duration:.2f} seconds")

epoch_details_df = pd.DataFrame(epoch_details)

# When switching to a new task, evaluate on previously learned tasks
if current_goal != "original" and current_goal not in previous_goals:
    previous_goals.append(current_goal)
    retention_scores = evaluate_knowledge_retention(q_network, env, previous_goals)
    for goal, score in retention_scores.items():
        retention_results[reg_type]["previous_goal_rewards"].setdefault(goal, []).append(score)

# Training time tracking
training_end_time = time.time()
print(f"Total Training Time: {training_end_time - training_start_time:.2f} seconds")

# Save trained model
torch.save(q_network.state_dict(), "dqn_acrobot_model.pth")

# --- Metrics and Plotting ---
# Improved Plotting of Average Convergence Speed per Task per Regularization Type
for reg_type in regularization_types:
    # Filter valid goals that have convergence data
    valid_goals = [goal for goal in goals + ["original"] if len(convergence_log[goal]) > 0]
    # Only keep goals with valid convergence data (not inf)
    # valid_goals = [goal for i, goal in enumerate(valid_goals) if avg_convergence_speeds[i] != float('inf')]

# Plot task-specific average rewards per regularization type
print("\n--- Evaluation Phase ---")
for reg_type in regularization_types:
    print(f"Evaluating model with Regularization Type: {reg_type}")

    eval_step = 0
    avg_rewards_per_goal = {}
    for goal in goals + ["original"]:
        # Evaluate agent and log action distribution
        avg_reward = evaluate_agent(env, q_network, num_episodes=5, goal=goal)

        # Save average rewards for each goal
        avg_rewards_per_goal[goal] = avg_reward
        results[reg_type]["retention_score"].append(avg_reward)
        eval_step += 1  # Increment eval step after each goal evaluation

    print(f"Avg Reward per Goal for {reg_type}: {avg_rewards_per_goal}")
env.close()

# --- Validation Phase ---
# Run the validation phase
print("\n--- Validation Phase ---")
validation_results = validate_agent(q_network, env, validation_goals, num_episodes=5)

# Log and display validation results
for goal, avg_reward in validation_results.items():
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

    return testing_results

# Call the structured_testing function to get results for each scenario
structured_testing_results = structured_testing(q_network, env, testing_scenarios, num_episodes=50, max_steps=500)

# Assuming structured_testing_results is the output of the function
for scenario_name, result in structured_testing_results.items():
    print(f"\n--- Results for Scenario: {scenario_name} ---")
    print(f"Average Reward: {result['average_reward']:.2f}")
    print(f"Average Duration (Steps): {result['average_duration']:.2f}")
    print(f"Standard Deviation of Reward: {result['std_reward']:.2f}")

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

print("\n--- Testing Phase ---")
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
    "epoch_details": epoch_details,
    "validation_results": validation_results,
    "episode_duration": episode_duration,
    "structured_testing_results": structured_testing_results,
    "testing_results": testing_results,
    "episode": range(window_size, episode_count + 1),
    "rolling_avg_reward": avg_reward,
    "rolling_std_reward": std_reward,
    "goal_performance_history": goal_performance_history,
}

all_results = []

def process_results(key, source_name, consolidated_results, all_results):
    """Process a given key and add corresponding results to all_results."""
    if key in consolidated_results:
        if isinstance(consolidated_results[key], (list, dict)):
            df = pd.DataFrame(consolidated_results[key])
            df['Source'] = source_name
            all_results.append(df)
        else:
            print(f"Unexpected data type for key '{key}', expected list or dict.")

# Define result keys and corresponding source names
result_keys = [
    ('epoch_details', 'Epoch Details'),  # Contains details about row A-L
    ('structured_testing_results', 'Structured Gradual Increase Testing Results'),
    ('testing_results', 'Noise Testing Results'),
    ('comparison_results', 'Comparison Results'),
    ('goal_performance_history', 'Goal Performance (every 50 episodes)'),
]

# Process and append each result type using the function
for key, source_name in result_keys:
    process_results(key, source_name, consolidated_results, all_results)

# Handle validation results (history and snapshot)
if validation_results_history:
    validation_results_flat = []
    for entry in validation_results_history:
        flat_entry = {'episode': entry['episode']}
        flat_entry.update(entry['validation_results'])
        validation_results_flat.append(flat_entry)
    validation_df = pd.DataFrame(validation_results_flat)
    validation_df['Source'] = 'Validation Results (every 50 episodes)'
    all_results.append(validation_df)

if 'validation_results' in consolidated_results and isinstance(consolidated_results['validation_results'], dict):
    validation_results_flat = []
    for goal, reward in consolidated_results['validation_results'].items():
        validation_results_flat.append({'goal': goal, 'average_reward': reward})
    validation_df = pd.DataFrame(validation_results_flat)
    validation_df['Source'] = 'Validation Results (Final Results)'
    all_results.append(validation_df)

if 'goal_performance_history' in locals() and goal_performance_history:
    goal_performance_flat = []
    for entry in goal_performance_history:
        flat_entry = {'episode': entry['episode']}
        flat_entry.update(entry)  # Update with goal performance data
        goal_performance_flat.append(flat_entry)
    goal_performance_df = pd.DataFrame(goal_performance_flat)
    goal_performance_df['Source'] = 'Goal Performance (every 50 episodes)'
    all_results.append(goal_performance_df)

# Concatenate all DataFrames into one final DataFrame
final_df = pd.concat(all_results, ignore_index=True)
filename = f"cartpole_results_{'rq1' if is_rq1 else 'rq2'}_{mode if mode != 'none' else f'fixed_{fixed_reg_type}'}.xlsx"
final_df.to_excel(filename, index=False)
print(f"All results saved to '{filename}'")

env.close()