# !pip install gymnasium
# !pip install numpy
# !pip install matplotlib
import time
import gymnasium as gym
import numpy as np

# Create the Acrobot environment using Gymnasium
env = gym.make('Acrobot-v1', render_mode='human')

# Parameters
switch_threshold = 10  # Step threshold to switch reward structures
noise_factor = 0.1  # Noise level for observation noise injection
num_episodes = 100  # Number of episodes to run

# Define bins for each state variable (adjusted for Acrobot)
n_bins = [6, 6, 6, 6, 6, 6]  # 6 bins for each of the 6 state components

# Initialize Q-table based on discretized state space (updated for Acrobot's action space of 3)
q_table = np.zeros(n_bins + [env.action_space.n])


# Discretize continuous state variables for Acrobot
def discretize_states(state, bins):
    # Adjusted state bounds for Acrobot's state space
    upper_bounds = [1, 1, 1, 1, 4 * np.pi, 9 * np.pi]
    lower_bounds = [-1, -1, -1, -1, -4 * np.pi, -9 * np.pi]

    ratios = [(state[i] - lower_bounds[i]) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]
    new_state = [int(round((bins[i] - 1) * ratios[i])) for i in range(len(state))]
    new_state = [min(bins[i] - 1, max(0, new_state[i])) for i in range(len(state))]

    return tuple(new_state)


# Function to inject noise into the observation
def inject_noise(state, noise_factor):
    state_array = np.array(state)  # Convert state to a NumPy array
    noise = np.random.randn(*state_array.shape) * noise_factor
    return state_array + noise  # Add noise to the state observations


# Strategy for balancing exploration and exploitation
def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore: random action
    else:
        return np.argmax(q_table[state])  # Exploit: action with the highest Q-value


# Q-learning update rule
def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    best_next_action = np.argmax(q_table[next_state])
    q_table[state][action] = q_table[state][action] + alpha * (
            reward + gamma * q_table[next_state][best_next_action] - q_table[state][action]
    )


# Training parameters
initial_epsilon = 1.0  # Start with 100% exploration
min_epsilon = 0.01  # Minimum exploration rate
decay_rate = 0.995  # Rate at which epsilon decays
epsilon = initial_epsilon
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor

# Running multiple episodes
for episode in range(num_episodes):
    state, info = env.reset()

    # Visualize the environment
    gym.make("Acrobot-v1", render_mode="rgb_array")
    env.render()
    time.sleep(1)

    state = inject_noise(state, noise_factor)
    state = discretize_states(state, n_bins)

    total_reward = 0
    adjusted_reward = 0

    for step in range(500):
        action = choose_action(state, epsilon)
        next_state, reward, done, truncated, info = env.step(action)

        # Apply noise and discretize the next state
        next_state_noisy = inject_noise(next_state, noise_factor)
        next_state_discrete = discretize_states(next_state_noisy, n_bins)

        # Task switching based on the step count (rewards change after switch_threshold steps)
        if step < switch_threshold:
            adjusted_reward = reward
        else:
            # Task 2: Modify the reward based on angles or other conditions
            adjusted_reward = reward  # Adjust this part based on custom logic if needed

        # Update the Q-table using Q-learning
        update_q_table(q_table, state, action, adjusted_reward, next_state_discrete, alpha, gamma)

        state = next_state_discrete
        total_reward += adjusted_reward

        print(f"Episode {episode + 1}, Step: {step}, Reward: {reward}, Adjusted Reward: {adjusted_reward}")

        if done or truncated:
            print(f"Episode {episode + 1} finished after {step + 1} steps with total reward: {total_reward}")
            break

    epsilon = max(min_epsilon, epsilon * decay_rate)

env.close()
