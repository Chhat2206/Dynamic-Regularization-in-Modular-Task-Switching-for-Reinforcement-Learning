!pip install gymnasium
!pip install numpy
!pip install matplotlib
!pip install stable-baselines3

import gymnasium as gym
import numpy as np
import random

# Parameters
alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor
epsilon = 0.1        # Exploration rate
num_episodes = 5000  # Number of episodes to train
num_discretization_bins = 6  # Number of bins for discretizing continuous state space

# Create the environment
env = gym.make('Acrobot-v1')

# Discretize the state space
def discretize_state(state):
    bins = [
        np.linspace(-1.0, 1.0, num_discretization_bins),  # cos(theta1)
        np.linspace(-1.0, 1.0, num_discretization_bins),  # sin(theta1)
        np.linspace(-1.0, 1.0, num_discretization_bins),  # cos(theta2)
        np.linspace(-1.0, 1.0, num_discretization_bins),  # sin(theta2)
        np.linspace(-4.0, 4.0, num_discretization_bins),  # theta1_dot
        np.linspace(-9.0, 9.0, num_discretization_bins),  # theta2_dot
    ]
    discretized_state = tuple(
        np.digitize(state[i], bins[i]) - 1 for i in range(len(state))
    )
    return discretized_state

# Initialize the Q-table
state_space_size = (num_discretization_bins,) * env.observation_space.shape[0]
action_space_size = env.action_space.n
Q = np.zeros(state_space_size + (action_space_size,))

# Epsilon-greedy action selection
def select_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

# SARSA Training
for episode in range(num_episodes):
    state = discretize_state(env.reset()[0])
    action = select_action(state)
    done = False
    total_reward = 0

    while not done:
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)
        next_action = select_action(next_state)

        # SARSA update rule
        Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])

        # Update state and action
        state = next_state
        action = next_action
        total_reward += reward

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Close the environment
env.close()
