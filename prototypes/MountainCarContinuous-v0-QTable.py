# Q-Table has terrible results for this
# !pip install gymnasium
# !pip install numpy
# !pip install matplotlib
# !pip install stable-baselines3

import time

import gymnasium as gym
import numpy as np

# Create the MountainCarContinuous environment using Gymnasium
env = gym.make('MountainCarContinuous-v0', render_mode='human')

# Parameters
switch_threshold = 10  # Step threshold to switch reward structures
noise_factor = 0.1  # Noise level for observation noise injection
num_episodes = 100  # Increased number of episodes to run

# Define bins for each state variable
n_bins = [6, 6]  # Example: 6 bins for each of the 2 state components. Modify as needed.

# Initialize Q-table based on discretized state space
q_table = np.zeros(n_bins + [3])  # For continuous actions, we need to adjust action bins accordingly

# Making continuous values into discrete
def discretize_states(state, bins):
    # Adjusted state bounds for MountainCarContinuous-v0
    upper_bounds = [0.6, 0.07]  # Position and velocity upper bounds
    lower_bounds = [-1.2, -0.07]  # Position and velocity lower bounds

    ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]
    new_state = [int(round((bins[i] - 1) * ratios[i])) for i in range(len(state))]
    new_state = [min(bins[i] - 1, max(0, new_state[i])) for i in range(len(state))]

    return tuple(new_state)

# Function to inject noise into the observation
def inject_noise(state, noise_factor):
    state_array = np.array(state)  # Convert state to a NumPy array
    noise = np.random.randn(*state_array.shape) * noise_factor
    return state_array + noise  # Add noise to the state observations

# Strategy for balancing exploration and exploitation for continuous actions
def choose_action_continuous(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return [np.random.uniform(-1, 1)]  # Explore: random continuous action
    else:
        return [np.argmax(q_table[state]) - 1]  # Exploit: based on discrete action in [-1, 0, 1] range

# Q-learning update rule
def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    best_next_action = np.argmax(q_table[next_state])
    q_table[state][int(action * 1 + 1)] = q_table[state][int(action * 1 + 1)] + alpha * (
        reward + gamma * q_table[next_state][best_next_action] - q_table[state][int(action * 1 + 1)]
    )

# Training parameters
initial_epsilon = 1.0  # Start with 100% exploration
min_epsilon = 0.01  # Minimum exploration rate
decay_rate = 0.995  # Rate at which epsilon decays
epsilon = initial_epsilon  # Set epsilon to start value before training
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor

# Running multiple episodes
for episode in range(num_episodes):
    state, info = env.reset()

    env.render()
    time.sleep(1)

    state = inject_noise(state, noise_factor)
    state = discretize_states(state, n_bins)

    total_reward = 0

    for step in range(500):
        action = choose_action_continuous(state, epsilon)

        # Perform the action in the environment
        next_state, reward, done, truncated, info = env.step(action)

        next_state_noisy = inject_noise(next_state, noise_factor)
        next_state_discrete = discretize_states(next_state_noisy, n_bins)

        # Task switching based on the step count
        if step < switch_threshold:
            adjusted_reward = reward
        else:
            position = next_state[0]
            velocity = next_state[1]

            if position > 0.5:
                adjusted_reward = 2 * reward
            else:
                adjusted_reward = reward - 1

            if abs(velocity) > 0.03:
                adjusted_reward -= 1

        # Update the Q-table using Q-learning
        update_q_table(q_table, state, action[0], adjusted_reward, next_state_discrete, alpha, gamma)

        state = next_state_discrete
        total_reward += adjusted_reward

        print(f"Episode {episode + 1}, Step: {step}, Reward: {reward}, Adjusted Reward: {adjusted_reward}")

        if done or truncated:
            print(f"Episode {episode + 1} finished after {step + 1} steps with total reward: {total_reward}")
            break

    epsilon = max(min_epsilon, epsilon * decay_rate)

# Close the environment after running all episodes
env.close()
