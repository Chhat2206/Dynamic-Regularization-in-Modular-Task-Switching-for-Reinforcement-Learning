import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define the DQN Model
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.001
BUFFER_SIZE = 1_000_000
BATCH_SIZE = 256
EPSILON_START = 1.0
EPSILON_END = 0.3
EPSILON_DECAY = 0.995
NOISE_STD_DEV = 0.1  # Standard deviation of Gaussian noise

# Set up environment and replay buffer
env = gym.make('Acrobot-v1', render_mode='human')  # Enable rendering
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

replay_buffer = deque(maxlen=BUFFER_SIZE)

# Instantiate model, loss, and optimizer
policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()


# Epsilon-greedy action selection
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            return policy_net(state).argmax().item()


# Training loop
def train_model():
    if len(replay_buffer) < BATCH_SIZE:
        return

    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    # Add Gaussian noise to states
    noise = np.random.normal(0, NOISE_STD_DEV, states.shape)
    states = states + torch.FloatTensor(noise).to(device)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0]
    target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = loss_fn(q_values, target_q_values.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Main training loop
num_episodes = 1000
epsilon = EPSILON_START

for episode in range(1, num_episodes + 1):
    state = env.reset()
    total_reward = 0

    for t in range(200):
        env.render()  # Render the environment for visualization
        action = select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)

        replay_buffer.append((state, action, reward, next_state, float(done)))
        state = next_state
        total_reward += reward

        train_model()

        if done:
            break

    # Decay epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # Update target network every few episodes
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Print progress
    noise_value = NOISE_STD_DEV
    print(f"Episode: {episode}, Reward: {total_reward}, Noise: {noise_value}, Epsilon: {epsilon:.2f}")

# Save the model
torch.save(policy_net.state_dict(), "dqn_acrobot_model_cuda.pth")

env.close()
