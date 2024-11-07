import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Create the environment
# env = gym.make("Acrobot-v1")
env = gym.make("Acrobot-v1", render_mode='Human')

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
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
min_epsilon = 0.01
batch_size = 64
replay_buffer_size = 50000
num_episodes = 500

# Initialize Q-network and target network
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
q_network = DQN(input_dim, output_dim)
target_network = DQN(input_dim, output_dim)
target_network.load_state_dict(q_network.state_dict())  # Copy weights to target network
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

# Replay buffer
replay_buffer = deque(maxlen=replay_buffer_size)


# Function to select action using epsilon-greedy policy
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # Exploration
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            q_values = q_network(state)
            return int(torch.argmax(q_values).item())  # Exploitation

# Function to add noise to state (optional, used during testing)
def add_noise_to_state(state, noise_level=0.1):
    noise = np.random.normal(0, noise_level, size=state.shape)
    return state + noise

# Function to train the DQN using replay buffer
def train_dqn():
    if len(replay_buffer) < batch_size:
        return

    # Sample random mini-batch
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
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

# -------------------- Training Phase --------------------

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = select_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)

        # Store experience in replay buffer
        replay_buffer.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        # Train the DQN
        train_dqn()

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Periodically update the target network
    if episode % 10 == 0:
        target_network.load_state_dict(q_network.state_dict())

    print(f"Training Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon}")

# Save the trained model
torch.save(q_network.state_dict(), "Acrobot/dqn_acrobot_model.pth")
print("Model saved successfully.")

# -------------------- Testing Phase --------------------

# Load the trained model for testing
q_network.load_state_dict(torch.load("Acrobot/dqn_acrobot_model.pth"))

# Define the number of episodes for testing
test_episodes = 20

# Testing loop (no training, only evaluation)
for episode in range(test_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        with torch.no_grad():  # Ensure no training happens during testing
            # Choose the best action (pure exploitation, no exploration)
            action = select_action(state, epsilon=0)

            # Perform the action in the environment
            next_state, reward, done, _, _ = env.step(action)

            # Add noise to the next state to test how the agent handles it
            next_state = add_noise_to_state(next_state, noise_level=0.1)

            # Move to the next state
            state = next_state
            total_reward += reward

    print(f"Test Episode: {episode + 1}, Total Reward: {total_reward}")

# Close the environment after testing
env.close()
