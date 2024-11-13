!pip install gymnasium

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

# -------------------- Parameters --------------------

# Create the environment
env = gym.make("Acrobot-v1", render_mode='Human')

# Set goals for custom task-switching
use_custom_goals = True  # Set False to train all episodes under the original goal in all episodes
goals = ["quick_recovery", "periodic_swing", "maintain_balance"]

# Hyperparameters
learning_rate = 0.0001  # Slightly lower for more stable learning, OG: 0.00025 is REALLY BAD
gamma = 0.99
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.996 # Slower decay from 0.995 to allow more exploration
min_epsilon = 0.01
batch_size = 256  # Recommended 128 or 256, prof says 32 has valuable research
replay_buffer_size = 100000  # Increased to allow more diverse experiences
replay_buffer = deque(maxlen=replay_buffer_size)
num_episodes = 500  # Increased number of episodes, OG: 500
target_update_frequency = 1000
convergence_threshold = -85
eval_interval = 50

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
        if reg_type == "batch_norm": self.bn1 = nn.ModuleList([nn.BatchNorm1d(128) for _ in range(num_modules)])
        self.fc2 = nn.Linear(128 * num_modules, 128)
        if reg_type == "batch_norm": self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, output_dim)
        if reg_type == "dropout": self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if len(x.shape) == 1: x = x.unsqueeze(0)  # Add a batch dimension, making it [1, input_dim]

        module_outputs = [torch.relu(layer(x)) for layer in self.fc1]
        if self.reg_type == "batch_norm": module_outputs = [self.bn1[i](output) for i, output in enumerate(module_outputs)]

        x = torch.cat(module_outputs, dim=-1)  # Concatenate outputs from all the modules
        if self.reg_type == "dropout": x = self.dropout(x)

        x = torch.relu(self.fc2(x))
        if self.reg_type == "batch_norm": x = self.bn2(x)

        return self.fc3(x)

# Function to assign rewards based on the goal
def get_goal_reward(reward, state, goal):
    if goal == "quick_recovery":
        if abs(state[0]) > 1.0: return reward + 1.0  # Reward the agent for quickly moving to a stable position from an extreme angle
    elif goal == "periodic_swing": return reward + np.sin(state[1] * 5.0)  # Reward swinging behavior
    elif goal == "maintain_balance": return reward + (1.0 - abs(state[0])) # Reward maintaining the top link at a horizontal position
    return reward

# Function to add noise to state for testing
def add_noise_to_state(state, noise_level=0.1):
    noise = np.random.normal(0, noise_level, size=state.shape)
    return state + noise

# Function to train the DQN using replay buffer
def train_dqn(q_network, target_network, optimizer, reg_type):
    if len(replay_buffer) < batch_size: return

    # Sample random mini-batch
    indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
    batch = [replay_buffer[i] for i in indices]
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    # Compute the Q-values for current states
    current_q_values = q_network(states).gather(1, actions)

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

# Evaluate agent on the given goal/environment
def evaluate_agent(env, num_episodes=5, goal="original"):
    total_rewards = []
    action_distribution = [0] * env.action_space.n  # To keep track of actions taken

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        max_eval_steps = 500

        while not done and step_count < max_eval_steps:
            with torch.no_grad():
                action = select_action(state, epsilon=0, q_network=q_network)
                next_state, reward, done, _, _ = env.step(action)
                total_reward += reward
                action_distribution[action] += 1
                state = next_state
                step_count += 1

        total_rewards.append(total_reward)
        print(f"Evaluation Episode: {episode + 1}, Goal: {goal}, Total Reward: {total_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"Average Reward after Evaluation for Goal '{goal}': {avg_reward}")
    print(f"Action Distribution during Evaluation: {action_distribution}")
    return avg_reward

# Initialize Q-network and target network
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

# For Pure Testing
# Define the regularization techniques to experiment with
regularization_types = ["dropout", "l1", "l2", "batch_norm"]

# Initialize an empty dictionary to store results for each regularization technique
results = {
    "dropout": {"rewards": [], "convergence_speed": [], "retention_score": [], "noise_resilience": []},
    "l1": {"rewards": [], "convergence_speed": [], "retention_score": [], "noise_resilience": []},
    "l2": {"rewards": [], "convergence_speed": [], "retention_score": [], "noise_resilience": []},
    "batch_norm": {"rewards": [], "convergence_speed": [], "retention_score": [], "noise_resilience": []}
}

# -------------------- Training DQN --------------------
# Function to select action using epsilon-greedy policy
def select_action(state, epsilon, q_network, noise_level=0.1):
    # Inject Guassian Noise to the state during training
    noisy_state = add_noise_to_state(state, noise_level) if noise_level > 0 else state
    if random.random() < epsilon:
        return env.action_space.sample()  # Exploration
    else:
        with torch.no_grad():
            noisy_state = torch.tensor(noisy_state, dtype=torch.float32)
            q_values = q_network(noisy_state)
            return int(torch.argmax(q_values).item())  # Exploitation

# -------------------- Training Loop --------------------

avg_rewards_per_100_episodes = []
original_phase_rewards = []
task_switching_phase_rewards = []

performance_log = {goal: {"rewards": [], "convergence_speed": []} for goal in goals}
performance_log["original"] = {"rewards": [], "convergence_speed": []}
convergence_log = {goal: [] for goal in goals}
window_size = 10  # Size for moving average window
task_switch_rewards = []
task_switch_count = 0
total_steps = 0

# Function to determine if agent has converged
def is_converged(rewards, threshold):
    if len(rewards) < window_size:
        return False
    return np.mean(rewards[-window_size:]) >= threshold


# After training, analyze metrics
for goal in performance_log:
    avg_reward = np.mean(performance_log[goal]["rewards"])
    avg_convergence_speed = np.mean(performance_log[goal]["convergence_speed"]) if performance_log[goal][
        "convergence_speed"] else None
    print(f"Goal: {goal}, Average Reward: {avg_reward}, Average Convergence Speed: {avg_convergence_speed}")

results = {
    reg_type: {
        "rewards": [],
        "convergence_speed": [],
        "retention_score": [],
        "noise_resilience": [],
        "task_rewards": {goal: [] for goal in goals}
    }
    for reg_type in regularization_types
}

# Wrapper for training and logging results for each regularization type
for reg_type in regularization_types:
    print(f"\nTraining with Regularization Type: {reg_type}")

    # Initialize a new model with the specified regularization type
    q_network = DQN(env.observation_space.shape[0], env.action_space.n, reg_type=reg_type)
    target_network = DQN(env.observation_space.shape[0], env.action_space.n, reg_type=reg_type)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate, weight_decay=1e-4 if reg_type == "l2" else 0) # Choose optimizer based on the regularization type
    replay_buffer = deque(maxlen=replay_buffer_size)
    episode_rewards = []  # Track rewards per episode
    avg_rewards_per_100_episodes = []
    convergence_episode = None  # To record the first episode of convergence
    task_switch_count = 0
    epsilon = 1.0

    # Training episodes with task switching
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        current_goal = "original" if not use_custom_goals or episode < 100 or episode >= num_episodes - 100 else random.choice(
            goals)
        if current_goal != "original":
            task_switch_count += 1
        while not done:
            action = q_network(torch.tensor(state, dtype=torch.float32)).argmax().item() if random.random() > epsilon else env.action_space.sample()
            next_state, reward, done, _, _ = env.step(action)
            reward = get_goal_reward(shape_reward(state, next_state, reward), state, current_goal)
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            train_dqn(q_network, target_network, optimizer, reg_type)
            if episode % target_update_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
        results[reg_type]["task_rewards"][current_goal].append(total_reward)

        # Evaluate agent after training
    avg_eval_reward = evaluate_agent(env, num_episodes=3, goal="original")
    results[reg_type]["rewards"].append(avg_eval_reward)

    # Calculate and store the average reward every 100 episodes
    if (episode + 1) % 100 == 0:
        avg_reward_last_100 = np.mean(episode_rewards[-100:])
        avg_rewards_per_100_episodes.append(avg_reward_last_100)
        print(f"Episode {episode + 1}: Average Reward (last 100 episodes): {avg_reward_last_100}")

# Training time tracking
training_end_time = time.time()
print(f"Total Training Time: {training_end_time - training_start_time:.2f} seconds")

# Save trained model
torch.save(q_network.state_dict(), "dqn_acrobot_model.pth")
with open("performance_log.pkl", "wb") as f:
    pickle.dump(results, f)

# Plot cumulative rewards per episode (fine-grained reward tracking)
plt.plot(episode_rewards, label='Episode Rewards')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Training Progress (Episode Rewards)')
plt.legend()
plt.show()

# Plot the 100-episode moving average to track performance trend over time
plt.figure(figsize=(10, 6))
plt.plot(avg_rewards_per_100_episodes, label="Average Reward per 100 Episodes")
plt.xlabel("100-Episode Intervals")
plt.ylabel("Average Reward")
plt.title("Average Reward Every 100 Episodes During Training")
plt.legend()
plt.show()



# Plot task-specific average rewards per regularization type
for reg_type in regularization_types:
    avg_task_rewards = {goal: np.mean(results[reg_type]["task_rewards"][goal]) for goal in goals}
    print(f"\n{reg_type} Summary - Avg Rewards per Goal: {avg_task_rewards}")

env.close()

# -------------------- Testing Phase --------------------

# Load the trained model for testing
q_network.load_state_dict(torch.load("dqn_acrobot_model.pth"))

# Define the number of episodes and maximum steps for testing
test_episodes = 20
max_steps = 500  # Limit to prevent long runs

# Timing for testing phase
testing_start_time = time.time()

# Define the number of episodes and maximum steps for testing
test_episodes = 20
max_steps = 500  # Limit to prevent long runs
eval_rewards = []

# Testing loop (no training, only evaluation against regular Acrobot goal)
for episode in range(test_episodes):
    episode_start_time = time.time()  # Start time for each test episode
    state, _ = env.reset()  # Initialize the environment and get the starting state
    total_reward = 0  # Initialize the total reward counter
    done = False  # Boolean to check if the episode is complete
    step_count = 0  # Step counter to limit maximum steps per episode

    while not done and step_count < max_steps:
        with torch.no_grad():  # Ensure no training happens during testing
            # Use the model to select an action without exploration
            action = select_action(state, epsilon=0, q_network=q_network, noise_level=0)  # Greedy policy
            next_state, reward, done, _, _ = env.step(action)

            # Optionally add noise to the next state (for testing noise resilience)
            noisy_state = add_noise_to_state(next_state, noise_level=0.4)
            state = noisy_state
            total_reward += reward
            step_count += 1

    # Track total rewards for evaluation
    eval_rewards.append(total_reward)
    episode_end_time = time.time()
    episode_duration = episode_end_time - episode_start_time
    print(f"Test Episode: {episode + 1}, Total Reward: {total_reward}, Duration: {episode_duration:.2f} seconds")

    # Evaluate and log every eval_interval episodes
    if (episode + 1) % eval_interval == 0:
        avg_reward = np.mean(eval_rewards)
        results[reg_type]["rewards"].append(avg_reward)  # Append to the specific reg_type rewards
        eval_rewards = []  # Reset eval_rewards for the next interval
        print(f"Evaluation after {episode + 1} episodes, Avg Reward: {avg_reward}")


# Calculate total testing time
testing_end_time = time.time()
testing_duration = testing_end_time - testing_start_time
print(f"Total Testing Time: {testing_duration:.2f} seconds")

# Close the environment after training
env.close()
