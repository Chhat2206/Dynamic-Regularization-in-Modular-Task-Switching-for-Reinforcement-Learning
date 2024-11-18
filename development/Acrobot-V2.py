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

# Create the environment
env = gym.make("Acrobot-v1")

# Set goals for custom task-switching
use_custom_goals = True  # Set False to train all episodes under the original goal in all episodes
goals = ["quick_recovery", "periodic_swing", "maintain_balance"]

# Hyperparameters

learning_rate = 0.0001  # Slightly lower for more stable learning, OG: 0.00025 is REALLY BAD
gamma = 0.99
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995 # Slower decay from 0.995 to allow more exploration
min_epsilon = 0.01
batch_size = 128  # Recommended 128 or 256, prof says 32 has valuable research
replay_buffer_size = 1000000  # Increased to allow more diverse experiences
replay_buffer = deque(maxlen=replay_buffer_size)
num_episodes = 500  # Increased number of episodes, OG: 500
target_update_frequency = 500
convergence_threshold = -100
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
        if self.reg_type == "batch_norm":
            module_outputs = [
                self.bn1[i](output) if output.size(0) > 1 else output
                for i, output in enumerate(module_outputs)
            ]

        x = torch.cat(module_outputs, dim=-1)  # Concatenate outputs from all the modules
        if self.reg_type == "dropout":
            x = self.dropout(x)

        x = torch.relu(self.fc2(x))
        if self.reg_type == "batch_norm" and x.size(0) > 1:  # Only apply batch norm if batch size > 1
            x = self.bn2(x)

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
    states = torch.tensor(np.array(states), dtype=torch.float32).to('cuda')
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to('cuda')
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to('cuda')
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to('cuda')
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to('cuda')

    # Compute the Q-values for current states
    current_q_values = q_network(states).gather(1, actions)

    # Compute the target Q-values
    with torch.no_grad():
        next_q_values = target_network(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * gamma * next_q_values

    # Compute loss
    loss = nn.MSELoss()(current_q_values, target_q_values.to('cuda'))

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
# regularization_types = ["dropout", "l1", "l2", "batch_norm"]
# regularization_types = ["dropout"]
# regularization_types = ["l1"]
# regularization_types = ["l2"]
# regularization_types = ["batch_norm"]

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

avg_rewards_per_100_episodes = []
original_phase_rewards = []
task_switching_phase_rewards = []

performance_log = {goal: {"rewards": [], "convergence_speed": []} for goal in goals}
performance_log["original"] = {"rewards": [], "convergence_speed": []}
window_size = 20
task_switch_rewards = []
task_switch_count = 0
total_steps = 0

results = {
    reg_type: {
        "rewards": [],
        "convergence_speed": [],
        "retention_score": [],
        "noise_resilience": [],
        "task_rewards": {goal: [] for goal in goals + ["original"]}
    }
    for reg_type in regularization_types
}

# Function to determine if agent has converged
def is_converged(rewards, threshold, window_size=10):
    if len(rewards) < window_size:
        return False
    return np.mean(rewards[-window_size:]) >= threshold

# Dictionary to track convergence speed for each task
convergence_log = {goal: [] for goal in goals + ["original"]}

# Number of episodes taken to converge after switching to a task
current_task_episode_count = {goal: 0 for goal in goals + ["original"]}

# Training episodes with task switching
for reg_type in regularization_types:
    print(f"\nTraining with Regularization Type: {reg_type}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

    # Initialize model, target, optimizer, and replay buffer
    q_network = DQN(env.observation_space.shape[0], env.action_space.n, reg_type=reg_type).to('cuda')
    target_network = DQN(env.observation_space.shape[0], env.action_space.n, reg_type=reg_type).to('cuda')
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate, weight_decay=1e-4 if reg_type == "l2" else 0)
    replay_buffer = deque(maxlen=replay_buffer_size)

    epsilon = 1.0
    episode_rewards = []  # Track rewards per episode

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        # Determine current task
        if not use_custom_goals or episode < 100 or episode >= num_episodes - 100:
            current_goal = "original"
        else:
            current_goal = random.choice(goals)

        # Reset convergence counter if switching to a new task
        if episode > 0 and current_goal != previous_goal:
            current_task_episode_count[current_goal] = 0
        previous_goal = current_goal

        # Run the episode
        while not done:
            action = q_network(torch.tensor(state, dtype=torch.float32).to('cuda')).argmax().item() if random.random() > epsilon else env.action_space.sample()
            next_state, reward, done, _, _ = env.step(action)
            reward = get_goal_reward(shape_reward(state, next_state, reward), state, current_goal)
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            train_dqn(q_network, target_network, optimizer, reg_type)

            # Update the target network periodically
            if total_steps % target_update_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
        results[reg_type]["task_rewards"][current_goal].append(total_reward)

        # Track the number of episodes taken to converge
        current_task_episode_count[current_goal] += 1

        # Check if the current task has converged using task-specific reward history
        if is_converged(results[reg_type]["task_rewards"][current_goal][-window_size:], convergence_threshold):
            convergence_log[current_goal].append(current_task_episode_count[current_goal])
            current_task_episode_count[current_goal] = 0  # Reset episode counter for next convergence tracking

        # Print progress every episode
        print(f"Episode: {episode + 1}/{num_episodes}, Goal: {current_goal}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    # After training, analyze convergence speed for each task
    for goal in convergence_log:
        if len(convergence_log[goal]) > 0:
            avg_convergence_speed = np.mean(convergence_log[goal])
        else:
            avg_convergence_speed = float('inf')  # Use a large number to indicate non-convergence
        print(
            f"Goal: {goal}, Average Convergence Speed: {avg_convergence_speed if avg_convergence_speed != float('inf') else 'Did not converge'} episodes")

# Training time tracking
training_end_time = time.time()
print(f"Total Training Time: {training_end_time - training_start_time:.2f} seconds")

# Save trained model
torch.save(q_network.state_dict(), "dqn_acrobot_model.pth")
with open("performance_log.pkl", "wb") as f:
    pickle.dump(results, f)

# Plotting average rewards per task per regularization type
for reg_type in regularization_types:
    plt.figure(figsize=(10, 6))
    for goal in goals + ["original"]:
        plt.plot(results[reg_type]["task_rewards"][goal], label=f"{goal} Rewards ({reg_type})")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title(f"Average Rewards per Task ({reg_type})")
    plt.legend()
    plt.show()

# Improved Plotting of Average Convergence Speed per Task per Regularization Type
for reg_type in regularization_types:
    # Filter valid goals that have convergence data
    valid_goals = [goal for goal in goals + ["original"] if len(convergence_log[goal]) > 0]
    avg_convergence_speeds = [np.mean(convergence_log[goal]) if len(convergence_log[goal]) > 0 else float('inf') for goal in valid_goals]

    # Only keep goals with valid convergence data (not inf)
    valid_goals = [goal for i, goal in enumerate(valid_goals) if avg_convergence_speeds[i] != float('inf')]
    avg_convergence_speeds = [speed for speed in avg_convergence_speeds if speed != float('inf')]

    # Only plot if there is data available
    if len(valid_goals) > 0:
        plt.figure(figsize=(10, 6))
        plt.bar(valid_goals, avg_convergence_speeds)
        plt.xlabel("Tasks")
        plt.ylabel("Average Episodes to Converge")
        plt.title(f"Convergence Speed per Task ({reg_type})")
        plt.xticks(rotation=45)
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.grid(axis='y')  # Add a grid along the y-axis for better visual interpretation
        plt.show()
    else:
        print(f"No valid convergence data for Regularization Type: {reg_type}")

# Plot task-specific average rewards per regularization type
print("\n--- Evaluation Phase ---")
for reg_type in regularization_types:
    print(f"Evaluating model with Regularization Type: {reg_type}")
    avg_rewards_per_goal = {}
    for goal in goals + ["original"]:
        avg_reward = evaluate_agent(env, num_episodes=5, goal=goal)
        avg_rewards_per_goal[goal] = avg_reward
        results[reg_type]["retention_score"].append(avg_reward)
    print(f"Avg Reward per Goal for {reg_type}: {avg_rewards_per_goal}")

env.close()

# -------------------- Testing Phase --------------------

# Define device (use GPU if available, otherwise fallback to CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the trained model for testing
q_network.load_state_dict(torch.load("dqn_acrobot_model.pth"))
q_network.to(device)  # Move the model to the correct device

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
        episode_start_time = time.time()  # Start time for each test episode
        state, _ = env.reset()  # Initialize the environment and get the starting state
        state = torch.tensor(state, dtype=torch.float32).to(device)  # Move state to the correct device
        total_reward = 0  # Initialize the total reward counter
        done = False  # Boolean to check if the episode is complete
        step_count = 0  # Step counter to limit maximum steps per episode

        while not done and step_count < max_steps:
            with torch.no_grad():  # Ensure no training happens during testing
                # Use the model to select an action without exploration
                action = select_action(state.cpu().numpy(), epsilon=0, q_network=q_network, noise_level=noise_level)  # Greedy policy
                next_state, reward, done, _, _ = env.step(action)

                # Optionally add noise to the next state (for testing noise resilience)
                next_state = add_noise_to_state(next_state, noise_level)
                state = torch.tensor(next_state, dtype=torch.float32).to(device)  # Move next state to the correct device
                total_reward += reward
                step_count += 1

        # Track total rewards and duration for evaluation
        eval_rewards.append(total_reward)
        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        episode_durations.append(episode_duration)

        # Print progress for each test episode with detailed info
        print(f"Test {episode + 1}/{test_episodes}: Noise Level {noise_level}, Total Reward: {total_reward:.2f}, "
              f"Duration: {episode_duration:.2f} seconds, Steps: {step_count}")

    # Calculate statistics
    avg_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    avg_duration = np.mean(episode_durations)

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
    print()

# Summary statistics for interpretation
print("\n--- Summary Statistics for Testing Phase ---")
print("Noise Levels Tested:", testing_results["noise_level"])
print("Average Rewards:", testing_results["average_reward"])
print("Standard Deviations of Rewards:", testing_results["std_reward"])
print("Average Episode Durations:", testing_results["average_duration"])

# Plotting Reward Curves for Different Regularization Techniques
plt.figure(figsize=(10, 6))
for reg_type in regularization_types:
    for goal in goals + ["original"]:
        plt.plot(results[reg_type]["task_rewards"][goal], label=f"{goal} ({reg_type})")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Reward Curves for Different Regularization Techniques")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()


# Plotting Average Convergence Speed per Task
for reg_type in regularization_types:
    valid_goals = [goal for goal in goals + ["original"] if len(convergence_log[goal]) > 0]
    avg_convergence_speeds = [np.mean(convergence_log[goal]) for goal in valid_goals]
    std_devs = [np.std(convergence_log[goal]) for goal in valid_goals]

    if len(valid_goals) > 0:
        plt.figure(figsize=(10, 6))
        plt.bar(valid_goals, avg_convergence_speeds, yerr=std_devs, capsize=5, alpha=0.7)
        plt.xlabel("Tasks")
        plt.ylabel("Average Episodes to Converge")
        plt.title(f"Convergence Speed per Task with Error Bars ({reg_type})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(axis='y')
        plt.show()
    else:
        print(f"No valid convergence data for Regularization Type: {reg_type}")


# Plotting testing results for different noise levels
plt.figure(figsize=(15, 5))

# Plot average rewards for each noise level
plt.subplot(1, 3, 1)
plt.plot(testing_results["noise_level"], testing_results["average_reward"], marker='o', linestyle='-', color='b', label='Avg Reward')
plt.xlabel("Noise Level")
plt.ylabel("Average Reward")
plt.title("Average Reward vs. Noise Level")
plt.legend()  # Add legend to indicate the plotted line
plt.grid(True)

# Plot standard deviation of rewards for each noise level
plt.subplot(1, 3, 2)
plt.plot(testing_results["noise_level"], testing_results["std_reward"], marker='o', linestyle='-', color='r', label='Std Dev of Reward')
plt.xlabel("Noise Level")
plt.ylabel("Reward Standard Deviation")
plt.title("Reward Std Dev vs. Noise Level")
plt.legend()  # Add legend to indicate the plotted line
plt.grid(True)

# Plot average episode duration for each noise level
plt.subplot(1, 3, 3)
plt.plot(testing_results["noise_level"], testing_results["average_duration"], marker='o', linestyle='-', color='g', label='Avg Duration')
plt.xlabel("Noise Level")
plt.ylabel("Average Duration (seconds)")
plt.title("Average Duration vs. Noise Level")
plt.legend()  # Add legend to indicate the plotted line
plt.grid(True)

plt.tight_layout()  # Ensure proper spacing between subplots
plt.show()

# Close the environment after testing
env.close()