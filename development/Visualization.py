import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time

# Create the environment
env = gym.make("Acrobot-v1", render_mode='human')  # Ensure human render mode

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
learning_rate = 0.0001
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.996
min_epsilon = 0.01
batch_size = 256
replay_buffer_size = 100000
num_episodes = 500
target_update_frequency = 1000

# Different Goals
goals = ["quick_recovery", "periodic_swing", "maintain_balance"]

# Initialize Q-network and target network
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
q_network = DQN(input_dim, output_dim)
target_network = DQN(input_dim, output_dim)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

# Replay buffer
replay_buffer = deque(maxlen=replay_buffer_size)

# Reward shaping function
def shape_reward(state, next_state, reward):
    if next_state[1] > state[1]:  # if the vertical position increases
        reward += 0.5
    return reward

# Function to select action using epsilon-greedy policy
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            q_values = q_network(state)
            return int(torch.argmax(q_values).item())

# Function to add noise to state for testing
def add_noise_to_state(state, noise_level=0.1):
    noise = np.random.normal(0, noise_level, size=state.shape)
    return state + noise
    # return state

# Load the trained model for testing
q_network.load_state_dict(torch.load("dqn_acrobot_model.pth"))

# Define the number of episodes and maximum steps for testing
test_episodes = 20
max_steps = 500

# Timing for testing phase
testing_start_time = time.time()

# Testing loop (no training, only evaluation against regular Acrobot goal)
for episode in range(test_episodes):
    episode_start_time = time.time()
    state, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0

    while not done and step_count < max_steps:
        env.render()  # Render the environment at each step
        with torch.no_grad():
            action = select_action(state, epsilon=0)
            next_state, reward, done, _, _ = env.step(action)

            # Optional: Add noise to the next state for robustness testing
            noisy_state = add_noise_to_state(next_state, noise_level=0.0)
            # 0.8 still is under 100
            # 2.0 brings it to 130 140ish

            # Use the noisy state or original state as needed
            state = noisy_state
            total_reward += reward
            step_count += 1

    episode_end_time = time.time()
    episode_duration = episode_end_time - episode_start_time
    print(f"Test Episode: {episode + 1}, Total Reward: {total_reward}, Duration: {episode_duration:.2f} seconds")

# Calculate total testing time
testing_end_time = time.time()
testing_duration = testing_end_time - testing_start_time
print(f"Total Testing Time: {testing_duration:.2f} seconds")

# Close the environment after testing
env.close()

# noise should be in the input or output or both (keep it in the input, sensor noise so you can test task switching & regularization to keep it very clean)4
# the most natural scanerio is that there is noise during training and testing, and if you are training something in the real world you have similar noise coniditions.
# focus only on sensor noise and for regularization you can focus on different techniques including those that inject noise into the parameter (parameter noise)
# should less muddy the waters and make the question more interesting