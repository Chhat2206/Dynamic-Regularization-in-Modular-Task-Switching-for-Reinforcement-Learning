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

# Neural network for DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
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
q_network = DQN(input_dim, output_dim)
target_network = DQN(input_dim, output_dim)
target_network.load_state_dict(q_network.state_dict())  # Copy weights to target network
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

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
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # Exploration
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            q_values = q_network(state)
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
def evaluate_agent(env, num_episodes=5):
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
        print(f"Evaluation Episode: {episode + 1}, Total Reward: {total_reward}")

    avg_reward = np.mean(total_rewards)  # Calculate average reward over evaluation episodes
    print(f"Average Reward after Evaluation: {avg_reward}")  # Print average reward after evaluation
    return avg_reward


# -------------------- Training Loop --------------------
total_steps = 0
performance_log = {goal: [] for goal in goals}
performance_log["original"] = []  # Includes the original goal

# Training metrics
episode_rewards = []
avg_rewards_per_100_episodes = []
convergence_threshold = -100
convergence_episode = None
convergence_check_interval = 100


for episode in range(num_episodes):
    episode_start_time = time.time()  # Track start time for each episode
    state, _ = env.reset()
    total_reward = 0
    done = False

    # Determine if we are in the original goal or task-switching phase
    if not use_custom_goals:
        current_goal = "original"  # Use the original Acrobot goal for all episodes
    elif episode < 100 or episode >= num_episodes - 100:
        current_goal = "original"  # Use the original Acrobot goal
    else:
        current_goal = random.choice(goals)  # Random task-switching phase

    while not done:
        action = select_action(state, epsilon)
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

    # Log total reward for the current goal
    performance_log[current_goal].append(total_reward)

    # Decay epsilon (exploration-exploitation balance)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Print episode details with timing
    episode_end_time = time.time()  # Track end time for each episode
    episode_duration = episode_end_time - episode_start_time
    print(
        f"Current Goal: {current_goal}, Training Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon}, Duration: {episode_duration:.2f} seconds")

    # Check for convergence every 100 episodes
    if episode % convergence_check_interval == 0 and episode > 0:
        avg_reward = np.mean(episode_rewards[-convergence_check_interval:])
        avg_rewards_per_100_episodes.append(avg_reward)
        if avg_reward > convergence_threshold and convergence_episode is None:
            convergence_episode = episode
            print(f"Convergence reached at episode: {convergence_episode}")

# Calculate total training time
training_end_time = time.time()
training_duration = training_end_time - training_start_time
print(f"Total Training Time: {training_duration:.2f} seconds")

# Save the trained model
torch.save(q_network.state_dict(), "dqn_acrobot_model.pth")
print("Model saved successfully.")

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