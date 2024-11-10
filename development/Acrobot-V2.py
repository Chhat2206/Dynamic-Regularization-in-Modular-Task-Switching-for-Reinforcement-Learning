import pickle

from matplotlib import pyplot as plt
!pip install gymnasium

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time

# Timing variables
training_start_time = time.time()

# -------------------- Parameters --------------------

# Create the environment
env = gym.make("Acrobot-v1", render_mode='Human')

# Configuration flag to choose between original or custom training
use_custom_goals = True  # Set False to train all episodes under the original goal in all episodes

# Choose regularization technique: "dropout", "l1", "l2", or None
reg_type = "dropout"
l1_weight_decay = 1e-4 if reg_type == "l2" else 0

# Neural network for DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, reg_type=None, num_modules=4):
        super(DQN, self).__init__()
        self.reg_type = reg_type
        self.num_modules = num_modules
        # Use nn.ModuleList to register the list of layers properly
        self.fc1 = nn.ModuleList([nn.Linear(input_dim, 128) for _ in range(num_modules)])
        self.fc2 = nn.Linear(128 * num_modules, 128)
        self.fc3 = nn.Linear(128, output_dim)

        if reg_type == "dropout":
            self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add a batch dimension, making it [1, input_dim]
        module_outputs = [torch.relu(layer(x)) for layer in self.fc1]
        x = torch.cat(module_outputs, dim=-1)  # Concatenate outputs from all the modules
        if self.reg_type == "dropout":
            x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Hyperparameters
learning_rate = 0.0001  # Slightly lower for more stable learning, OG: 0.00025 is REALLY BAD
gamma = 0.99
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.996 # Slower decay from 0.995 to allow more exploration
min_epsilon = 0.01
batch_size = 256  # Recommended 128 or 256, prof says 32 has valuable research
replay_buffer_size = 100000  # Increased to allow more diverse experiences
num_episodes = 500  # Increased number of episodes, OG: 500
target_update_frequency = 1000

# Different Goals
goals = ["quick_recovery", "periodic_swing", "maintain_balance"]

# Initialize Q-network and target network
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

q_network = DQN(input_dim, output_dim, reg_type="None")
target_network = DQN(input_dim, output_dim)
target_network.load_state_dict(q_network.state_dict())  # Copy weights to target network
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate, weight_decay=l1_weight_decay) # Weight decay with L2 regularization

# Replay buffer
replay_buffer = deque(maxlen=replay_buffer_size)

# Reward shaping function
def shape_reward(state, next_state, reward):
    # Reward the agent for increasing the height of the tip of the second link
    if next_state[1] > state[1]:  # if the vertical position increases
        reward += 0.5  # Provide a small positive reward for progress
    return reward

# -------------------- Training DQN --------------------

# Function to train the DQN using replay buffer
def train_dqn():
    if len(replay_buffer) < batch_size:
        return

    # Sample random mini-batch
    indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
    batch = [replay_buffer[i] for i in indices]
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)  # Convert to a single array first
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    # Compute the Q-values for current states
    current_q_values = q_network(states).gather(1, actions)

    # Compute the target Q-values
    with torch.no_grad():
        next_q_values = target_network(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * gamma * next_q_values

    # Compute loss
    loss = nn.MSELoss()(current_q_values, target_q_values)

    if reg_type == "l1":
        l1_loss = 0
        for param in q_network.parameters():
            l1_loss += torch.abs(param).sum()
        loss += l1_weight_decay * l1_loss

    # Update the network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Function to assign rewards based on the goal
def get_goal_reward(reward, state, goal):
    if goal == "quick_recovery":
        # Reward the agent for quickly moving to a stable position from an extreme angle
        if abs(state[0]) > 1.0:  # Example condition for extreme angle
            return reward + 1.0  # Reward for recovery
    elif goal == "periodic_swing":
        # Encourage oscillation behavior
        return reward + np.sin(state[1] * 5.0)  # Reward swinging behavior
    elif goal == "maintain_balance":
        # Reward maintaining the top link at a horizontal position
        return reward + (1.0 - abs(state[0]))  # More reward closer to horizontal

    return reward

# Function to select action using epsilon-greedy policy
def select_action(state, epsilon, noise_level=0.1):
    # Inject Guassian Noise to the state during training
    noisy_state = add_noise_to_state(state, noise_level) if reg_type == "noise" else state
    if random.random() < epsilon:
        return env.action_space.sample()  # Exploration
    else:
        with torch.no_grad():
            noisy_state = torch.tensor(noisy_state, dtype=torch.float32)
            q_values = q_network(noisy_state)
            return int(torch.argmax(q_values).item())  # Exploitation

# Function to add noise to state for testing
def add_noise_to_state(state, noise_level=0.1):
    noise = np.random.normal(0, noise_level, size=state.shape)
    return state + noise

# -------------------- Reward Shaping --------------------

# Compare the height of the pendilum's original height to the current height for simple rewarding of the feedback loop
def shape_reward(state, next_state, reward):
    if next_state[1] > state[1]:
        reward += 0.5

    return reward

# -------------------- Evaluate Agent --------------------

# Evaluate agent on the given goal/environment
def evaluate_agent(env, num_episodes=5, goal="original"):
    total_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()  # Reset the environment for each evaluation episode
        total_reward = 0
        done = False
        step_count = 0
        max_eval_steps = 500  # Limit the number of steps for evaluation

        while not done and step_count < max_eval_steps:
            with torch.no_grad():  # Ensures no training is happening while evaluating performance
                action = select_action(state, epsilon=0)  # Use greedy policy during evaluation (no exploration)
                next_state, reward, done, _, _ = env.step(action)
                total_reward += reward
                state = next_state
                step_count += 1

        total_rewards.append(total_reward)  # Log total reward for the episode
        # Add print statement for progress tracking
        print(f"Evaluation Episode: {episode + 1}, Goal: {goal}, Total Reward: {total_reward}")

    avg_reward = np.mean(total_rewards)  # Calculate average reward over evaluation episodes
    print(f"Average Reward after Evaluation for Goal '{goal}': {avg_reward}")  # Print average reward after evaluation
    return avg_reward

# -------------------- Training Loop --------------------

# Training metrics
episode_rewards = []
avg_rewards_per_100_episodes = []
original_phase_rewards = []
task_switching_phase_rewards = []
convergence_threshold = -85
convergence_episode = None
performance_log = {goal: {"rewards": [], "convergence_speed": []} for goal in goals}
performance_log["original"] = {"rewards": [], "convergence_speed": []}
convergence_log = {goal: [] for goal in goals}
window_size = 10  # Size for moving average window
eval_interval = 50  # Evaluate the agent every 50 episodes
task_switch_rewards = []
task_switch_count = 0
total_steps = 0

# Function to determine if agent has converged
def is_converged(rewards, threshold):
    if len(rewards) < window_size:
        return False
    return np.mean(rewards[-window_size:]) >= threshold

# Training metrics
for episode in range(num_episodes):
    episode_start_time = time.time()  # Track start time for each episode
    state, _ = env.reset()
    total_reward = 0
    done = False
    steps_to_converge = 0

    # Determine if we are in the original goal or task-switching phase
    if not use_custom_goals:
        current_goal = "original"  # Use the original Acrobot goal for all episodes
    elif episode < 100 or episode >= num_episodes - 100:
        current_goal = "original"  # Use the original Acrobot goal
    else:
        current_goal = random.choice(goals)  # Random task-switching phase

    if current_goal != "original":
        task_switch_count += 1

    while not done:
        action = select_action(state, epsilon, noise_level=0.1)
        next_state, reward, done, _, _ = env.step(action)

        # Adjust reward if not in the original goal phase
        if current_goal != "original":
            reward = get_goal_reward(reward, state, current_goal)

        reward = shape_reward(state, next_state, reward)

        # Store experience in replay buffer
        replay_buffer.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward
        total_steps += 1

        # Train the DQN
        train_dqn()

        # Update target network periodically
        if total_steps % target_update_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

    # Log performance metrics
    performance_log[current_goal]["rewards"].append(total_reward)
    if total_reward >= convergence_threshold:
        performance_log[current_goal]["convergence_speed"].append(steps_to_converge)
    if is_converged(performance_log[current_goal]["rewards"], convergence_threshold):
        convergence_log[current_goal].append(episode)

    if current_goal != "original":
        task_switch_rewards.append(total_reward)

    # Decay epsilon (exploration-exploitation balance)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Periodic evaluation every eval_interval episodes
    if episode % eval_interval == 0:
        avg_reward = evaluate_agent(env, num_episodes=3, goal=current_goal)
        performance_log[current_goal].setdefault("evaluation_rewards", []).append(avg_reward)

    # Print episode details
    episode_end_time = time.time()
    episode_duration = episode_end_time - episode_start_time
    print(
        f"Current Goal: {current_goal}, Training Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon}, Duration: {episode_duration:.2f} seconds")

# After training, analyze metrics
for goal in performance_log:
    avg_reward = np.mean(performance_log[goal]["rewards"])
    avg_convergence_speed = np.mean(performance_log[goal]["convergence_speed"]) if performance_log[goal][
        "convergence_speed"] else None
    print(f"Goal: {goal}, Average Reward: {avg_reward}, Average Convergence Speed: {avg_convergence_speed}")

# Calculate total training time
training_end_time = time.time()
training_duration = training_end_time - training_start_time
print(f"Total Training Time: {training_duration:.2f} seconds")

# Save the trained model
torch.save(q_network.state_dict(), "dqn_acrobot_model.pth")
print("Model saved successfully.")

# Save performance metrics
with open("performance_log.pkl", "wb") as f:
    pickle.dump(performance_log, f)
print("Performance log saved successfully.")

# -------------------- Testing Phase --------------------

# Load the trained model for testing
q_network.load_state_dict(torch.load("dqn_acrobot_model.pth"))

# Define the number of episodes and maximum steps for testing
test_episodes = 20
max_steps = 500  # Limit to prevent long runs

# Timing for testing phase
testing_start_time = time.time()

# Testing loop (no training, only evaluation against regular Acrobot goal)
for episode in range(test_episodes):
    episode_start_time = time.time()  # Start time for each test episode

    state, _ = env.reset()  # Initialize the environment and get the starting state
    total_reward = 0  # Initialize the total reward counter
    done = False  # Boolean to check if the episode is complete
    step_count = 0  # Step counter to limit maximum steps per episode

    while not done and step_count < max_steps:
        with torch.no_grad():  # Ensure no training happens during testing
            # Use the model to select an action without exploration (epsilon = 0)
            action = select_action(state, epsilon=0)
            # Take the action and get the next state, reward, and done status
            next_state, reward, done, _, _ = env.step(action)

            # Add noise to the next state to test how the agent handles it
            noisy_state = add_noise_to_state(next_state, noise_level=0.4)

            # Move to the next noisy state instead of the original one
            state = noisy_state  # Assign the noisy state as the new state
            total_reward += reward  # Accumulate the reward
            step_count += 1  # Increment step counter

    episode_end_time = time.time()  # End time for each test episode
    episode_duration = episode_end_time - episode_start_time
    print(f"Test Episode: {episode + 1}, Total Reward: {total_reward}, Duration: {episode_duration:.2f} seconds")

# Calculate total testing time
testing_end_time = time.time()
testing_duration = testing_end_time - testing_start_time
print(f"Total Testing Time: {testing_duration:.2f} seconds")

# Close the environment after testing
env.close()

# -------------------- Visualize Metrics --------------------

# Plot rewards per episode
plt.plot(episode_rewards, label='Episode Rewards')
plt.plot(avg_rewards_per_100_episodes, label='Average Rewards (100 episodes)')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.legend()
plt.show()

# Knowledge retention visualization
for goal in performance_log:
    if "evaluation_rewards" in performance_log[goal]:
        plt.plot(performance_log[goal]["evaluation_rewards"], label=f"{goal} Evaluation Rewards")
plt.xlabel('Evaluation Interval')
plt.ylabel('Average Reward')
plt.title('Knowledge Retention across Goals')
plt.legend()
plt.show()
