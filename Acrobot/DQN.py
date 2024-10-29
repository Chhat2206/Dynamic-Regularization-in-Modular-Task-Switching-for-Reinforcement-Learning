!pip install gymnasium

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Acrobot state vector explained (indices 0 to 5):

# state[0]: Cosine of the angle of the first link (cos(theta1))
# - Represents the horizontal position of the first link.
# - Ranges from -1 to 1.

# state[1]: Sine of the angle of the first link (sin(theta1))
# - Represents the vertical position of the first link.
# - Ranges from -1 to 1.

# state[2]: Cosine of the angle of the second link (cos(theta2))
# - Represents the horizontal position of the second link.
# - Helps determine how the second link is swinging relative to the first.
# - Ranges from -1 to 1.

# state[3]: Sine of the angle of the second link (sin(theta2))
# - Represents the vertical position of the second link.
# - Helps complete the description of the second link's position.
# - Ranges from -1 to 1.

# state[4]: Angular velocity of the first link (theta1_dot)
# - Represents how fast the first link is rotating.
# - Positive values indicate one direction of rotation, negative values indicate the other.

# state[5]: Angular velocity of the second link (theta2_dot)
# - Represents how fast the second link is rotating.
# - Positive values indicate one direction of rotation, negative values indicate the other.

# Cosine and Sine values (state[0] to state[3]) describe the position of the two links in terms of a circle,
# while angular velocities (state[4] and state[5]) describe how fast they are moving.

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
learning_rate = 0.0005  # Slightly lower for more stable learning, OG: 0.00025 is REALLY BAD
gamma = 0.99
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995 # Slower decay from 0.995 to allow more exploration
min_epsilon = 0.01
batch_size = 128  # Recommended 128 or 256, prof says 32 has valuable research
replay_buffer_size = 100000  # Increased to allow more diverse experiences
num_episodes = 300  # Increased number of episodes, OG: 500

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

# -------------------- Parameter Noise --------------------

# Adding parameter noise so that there are noise in the weights & it explores the different posibilities of the search space
def adds_parameter_noise(noise_level=0.1): #0.05->0.04. Once reward is increased, increase noise level
    with torch.no_grad():
        for param in q_network.parameters():
            param.add_(torch.randn_like(param) * noise_level)

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

performance_log = {goal: [] for goal in goals}
# performance_log = {
#     "quick_recovery": [],
#     "periodic_swing": [],
#     "maintain_balance": []
# }
performance_log["original"] = []  # Includes the original goal

for episode in range(num_episodes):
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
        # Adding parameter noise to encourage exploration
        adds_parameter_noise(noise_level=0.1)

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

        # Train the DQN
        train_dqn()

    # Log total reward for the current goal
    performance_log[current_goal].append(total_reward)

    # Decay epsilon (exploration-exploitation balance)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Periodically update the target network
    # if episode % 10 == 0:
    #     target_network.load_state_dict(q_network.state_dict())
    #
    # # Evaluate agent periodically to check learning progress on the current goal
    # if episode % 50 == 0:
    #     avg_reward = evaluate_agent(env, num_episodes=5)  # Evaluate on the current environment
    #     print(f"Evaluation after episode {episode + 1}: Average Reward on Current Goal ({current_goal}): {avg_reward}")
    #
    # # Evaluate agent periodically on all goals to track knowledge retention
    # if episode % 100 == 0 and episode > 0:  # Evaluate every 100 episodes after some progress
    #     for goal in goals + ["original"]:
    #         avg_reward = evaluate_agent(env, num_episodes=5)  # Use the evaluation function
    #         print(f"Evaluation after episode {episode + 1}: Average Reward on Goal '{goal}': {avg_reward}")

    # Print episode details
    print(f"Current Goal: {current_goal}, Training Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon}")

# Save the trained model
torch.save(q_network.state_dict(), "dqn_acrobot_model.pth")
print("Model saved successfully.")

# -------------------- Testing Phase --------------------

# Load the trained model for testing
q_network.load_state_dict(torch.load("dqn_acrobot_model.pth", weights_only=True))

# Define the number of episodes and maximum steps for testing
test_episodes = 20
max_steps = 500  # Limit to prevent long runs

# Function to add noise to state
def add_noise_to_state(state, noise_level=0.1):
    noise = np.random.normal(0, noise_level, size=state.shape) # Adds standard Guassian Noise
    return state + noise

# Testing loop (no training, only evaluation against regular Acrobot goal)
for episode in range(test_episodes):
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
            noisy_state = add_noise_to_state(next_state, noise_level=0.4) #

            # Print noisy and original states for debugging
            # print(f"Original: {state}, Noisy: {noisy_state}")

            # Move to the next noisy state instead of the original one
            state = noisy_state  # Assign the noisy state as the new state
            total_reward += reward  # Accumulate the reward
            step_count += 1  # Increment step counter

    print(f"Test Episode: {episode + 1}, Total Reward: {total_reward}")

    # Ensure environment closes after each episode to reset rendering issues
    # env.close()
    # env gym.make("Acrobot-v1", render_mode='Human')  # Reinitialize  reset rendering issues

# Close the environment after testing
env.close()