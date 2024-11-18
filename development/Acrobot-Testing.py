import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import time

# Device configuration (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the DQN network structure
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, num_modules=4):
        super(DQN, self).__init__()
        self.fc1 = nn.ModuleList([nn.Linear(input_dim, 128) for _ in range(num_modules)])
        self.fc2 = nn.Linear(128 * num_modules, 128)  # Adjust for concatenated outputs
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        if len(x.shape) == 1:  # Add a batch dimension if input is a single state
            x = x.unsqueeze(0)
        # Process input through each module in fc1
        module_outputs = [torch.relu(layer(x)) for layer in self.fc1]
        x = torch.cat(module_outputs, dim=-1)  # Concatenate outputs
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Function to add noise to the state
def add_noise_to_state(state, noise_level=0.1):
    noise = np.random.normal(0, noise_level, size=state.shape)
    return state + noise

# Function to select an action using the trained Q-network
def select_action(state, q_network, noise_level=0.0):
    noisy_state = add_noise_to_state(state, noise_level)
    noisy_state = torch.tensor(noisy_state, dtype=torch.float32).to(device)
    with torch.no_grad():
        q_values = q_network(noisy_state)
        action = int(torch.argmax(q_values).item())
    return action

# Load the environment
env = gym.make("Acrobot-v1")

# Load the trained model
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
q_network = DQN(input_dim, output_dim).to(device)
q_network.load_state_dict(torch.load("dqn_acrobot_model.pth"))
q_network.eval()

# Testing parameters
test_episodes = 50
max_steps = 500
noise_levels = [0.0, 0.1, 0.2, 0.4]  # Different levels of noise for testing

# Initialize results dictionary
testing_results = {
    "noise_level": [],
    "average_reward": [],
    "std_reward": [],
    "average_duration": []
}

# Start testing
testing_start_time = time.time()

for noise_level in noise_levels:
    print(f"\nTesting with Noise Level: {noise_level}")
    eval_rewards = []
    episode_durations = []

    for episode in range(test_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        episode_start_time = time.time()

        while not done and step_count < max_steps:
            action = select_action(state, q_network, noise_level=noise_level)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward
            step_count += 1

        episode_duration = time.time() - episode_start_time
        eval_rewards.append(total_reward)
        episode_durations.append(episode_duration)

        print(f"Episode {episode + 1}/{test_episodes} - Total Reward: {total_reward:.2f}, Duration: {episode_duration:.2f}s")

    avg_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    avg_duration = np.mean(episode_durations)

    testing_results["noise_level"].append(noise_level)
    testing_results["average_reward"].append(avg_reward)
    testing_results["std_reward"].append(std_reward)
    testing_results["average_duration"].append(avg_duration)

    print(f"Noise Level {noise_level} - Avg Reward: {avg_reward:.2f}, Std Reward: {std_reward:.2f}, Avg Duration: {avg_duration:.2f}s")

testing_end_time = time.time()
total_testing_duration = testing_end_time - testing_start_time
print(f"\nTotal Testing Time: {total_testing_duration:.2f} seconds")

# Plot results
plt.figure(figsize=(15, 5))

# Plot average rewards for each noise level
plt.subplot(1, 3, 1)
plt.plot(testing_results["noise_level"], testing_results["average_reward"], marker="o", linestyle="-", color="b")
plt.xlabel("Noise Level")
plt.ylabel("Average Reward")
plt.title("Average Reward vs. Noise Level")
plt.grid(True)

# Plot standard deviation of rewards for each noise level
plt.subplot(1, 3, 2)
plt.plot(testing_results["noise_level"], testing_results["std_reward"], marker="o", linestyle="-", color="r")
plt.xlabel("Noise Level")
plt.ylabel("Reward Standard Deviation")
plt.title("Reward Std Dev vs. Noise Level")
plt.grid(True)

# Plot average episode duration for each noise level
plt.subplot(1, 3, 3)
plt.plot(testing_results["noise_level"], testing_results["average_duration"], marker="o", linestyle="-", color="g")
plt.xlabel("Noise Level")
plt.ylabel("Average Duration (seconds)")
plt.title("Average Duration vs. Noise Level")
plt.grid(True)

plt.tight_layout()
plt.show()

# Close the environment
env.close()
