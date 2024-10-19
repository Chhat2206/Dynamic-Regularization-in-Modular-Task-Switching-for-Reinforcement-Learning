# !pip install gymnasium
# !pip install numpy
# !pip install matplotlib
import time

import gymnasium as gym
import numpy as np

# Create the CartPole environment using Gymnasium
# env = gym.make('Acrobot-v1')
env = gym.make('Acrobot-v1', render_mode='human')

# Parameters
switch_threshold = 10  # Step threshold to switch reward structures
noise_factor = 0.1  # Noise level for observation noise injection
num_episodes = 100  # Increased number of episodes to run

# Initializing the Q-table: (number of states, number of actions)
# Cartpole doesn't have a discrete observation space, so this is recommended discreting metrics
# Number of discrete states for each dimension (position, velocity, angle, angular velocity)

# Define bins for each state variable
n_bins = [6, 6, 6, 6]  # Example: 6 bins for each of the 4 state components. Modify as I go along

# Initialize Q-table based on discretized state space
q_table = np.zeros(n_bins + [env.action_space.n])
# q_table = np.zeros(...) creates an empty table of estimated expected future values.
# n_bins is a list that defines the number of discrete bins for each state variable. (i.e. 6 6 6 6)
# + [env.action_space.n] is the actions, so 2
# q_table.shape = (6, 6, 6, 6, 2)
# So if State: [2, 3, 1, 4] and action = 0 (left), then
# q_value = q_table[2, 3, 1, 4, 0]

# Making continuous values into discrete
def discretize_states(state, bins):
    # high gives the upper values of the state
    # = [gives highest possible value], 0.5 (all values above 0.5 are treated as 0.5 to avoid screwing over the table)
    # cart position, cart velocity pole angle, pole angle velocity
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], 0.5]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -0.5]

    # Current value of the state [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    # lower_bounds = min value, upper_bounds = max value
    # abs turns it positive for normalization
    # then state[i] makes the bound from 0 0.5 and 1
    # each state (1/4) has its own lower and upper bounds to run into this formula
    ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]

    # ratios is the normalized value
    # if you have 6 bins, it’s like having 6 intervals (labeled 0, 1, 2, 3, 4, and 5).
    # Example: Each bin could represent a time interval like 1 second.
    # This means the value is between bin 3 and bin 4.
    #
    #     Bin 3 might represent 3 seconds.
    #     Bin 4 might represent 4 seconds.
    #     The value you are looking at is 3.5 seconds (halfway between 3 and 4).
    # So now when you reach 3s, you can execute this action
    # round((bins[i] - 1) * ratios[i]):
    #     The round() function rounds the result to the nearest integer, giving us the closest bin. ex. 2.5 -> 3
    # for i in range(len(state)): This loops over each component of the state (e.g., cart position, velocity) and performs the above operation for each one.
    new_state = [int(round((bins[i] - 1) * ratios[i])) for i in range(len(state))]

    # max(0, new_state[i]):
    #
    #     This ensures that the bin index is at least 0.
    #     If new_state[i] is less than 0, it will be clamped to 0. This ensures that the bin index doesn’t go below the minimum valid value (which is bin 0).
    #
    # Using min(bins[i] - 1, new_state[i]) ensures that the bin index:
    #     Never exceeds the highest valid bin index (preventing an out-of-bounds error when accessing the Q-table).
    #     Keeps the state properly within the range of valid bin indices.
    new_state = [min(bins[i] - 1, max(0, new_state[i])) for i in range(len(state))]

    # the discretized state (which was a list of bin indices) is converted into a tuple.
    # it’s common to use a tuple as the key in a Q-table because tuples are immutable and can be easily used as keys in a dictionary
    return tuple(new_state)

# Function to inject noise into the observation
def inject_noise(state, noise_factor):
    state_array = np.array(state)  # Convert state to a NumPy array
    noise = np.random.randn(*state_array.shape) * noise_factor
    return state_array + noise  # Add noise to the state observations


# Strategy for balancing exploration and exploitation
def choose_action(state, epsilon):
    # np.random.uniform(0, 1) generates a random number between 0 and 1.
    # epsilon is a parameter that controls how often the agent should explore.
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore: random action
    else:
        return np.argmax(q_table[state])  # Exploit: action with the highest Q-value

# Q-learning update rule
# q_table: This is a table where the agent keeps track of the expected rewards for different actions in each state. It stores the agent's "knowledge" of what actions to take.
# state: The current situation the agent is in (e.g., the position of a cart in CartPole).
# action: The choice the agent made while in the current state (e.g., move left or right).
# reward: The immediate feedback the agent gets after taking the action (e.g., positive if it keeps the pole balanced, negative if it fails).
# next_state: The new situation the agent ends up in after taking the action (e.g., the new position of the cart).
# alpha: The learning rate, which decides how much the agent should adjust its previous knowledge based on new information. A higher value means it learns more from new experiences.
# gamma: The discount factor, which tells the agent how important future rewards are compared to immediate rewards. A higher value means the agent cares more about long-term rewards than short-term gains.
def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):

    # q_table[next_state], you get the Q-values for both actions (left and right) in the next state.
    # q_table = (2, 1, -0.1, 0.1): [4.5, 3.2],  # Q-values for [left, right], so it chooses left (4.5) cause its better
    # argmax returns the highest value, so 4.5
    best_next_action = np.argmax(q_table[next_state])

    # q_table[state][action]: This is the Q-value for the current state and the action the agent took.
    # alpha is the learning rate, a number between 0 and 1,  If alpha = 0.1, the Q-value will change by 10% of the new information.
    q_table[state][action] = q_table[state][action] + alpha * (
        # reward: The immediate reward
        # gamma is the discount factor (a number between 0 and 1) that determines how much future rewards matter.
        # best_next_action is expected future reward
        # Multiplying by gamma makes sure the future rewards are discounted (valued a bit less than immediate rewards).
        # so this entire thing is the total expected reward
        reward + gamma * q_table[next_state][best_next_action] - q_table[state][action]
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
    # Reset the environment to start a new episode
    state, info = env.reset()

    # Visualize the environment
    gym.make("CartPole-v1", render_mode="rgb_array")
    env.render()
    time.sleep(1)  # Adjust this to control the speed of the rendering

    state = inject_noise(state, noise_factor) # Applying noise to the initial state
    state = discretize_states(state, n_bins) # Converts continuous values into discrete categories (bins) to make them easier to manage.

    total_reward = 0  # Track total reward in each episode
    adjusted_reward = 0  # Initialize the adjusted reward

    for step in range(500):  # Steps per episode
        # Strategy to balancing exploration and exploitation
        action = choose_action(state, epsilon)

        # Perform the action in the environment
        next_state, reward, done, truncated, info = env.step(action)
        # env.step(action): This is a method that tells the environment to execute the specified action.
        # next state is the current situation of the enviorment to decide what to do next
        # reward is immediate
        # done checks if the episode is over
        # truncated checks if the episode ends due to te time limit, confirms that its not because the agent failed

        # Apply noise and discretize the next state
        next_state_noisy = inject_noise(next_state, noise_factor)
        next_state_discrete = discretize_states(next_state_noisy, n_bins)

        # Task switching based on the step count (rewards change after switch_threshold steps)
        if step < switch_threshold:
            # Task 1: Use the standard reward (no changes)
            adjusted_reward = reward
        else:
            # Task 2: Modify the reward based on the pole's angle (state[2] is the pole angle)
            pole_angle = next_state[2]  # Get the pole angle from the state
            cart_position = next_state[0] # Get the cart's position

            if abs(pole_angle) < 0.1:
                # Give a higher reward if the pole is near upright
                adjusted_reward = 2 * reward
            else:
                # Penalize if the pole is at a steep angle (about to fall)
                adjusted_reward = reward - 1

            # Add penalty for moving too far from the center (cart position)
            if abs(cart_position) > 1.0: # Threshold for cart position, adjust as needed
                adjusted_reward -= 1

        # Update the Q-table using Q-learning
        update_q_table(q_table, state, action, adjusted_reward, next_state_discrete, alpha, gamma)

        # Move to the next state
        state = next_state_discrete

        # Accumulate the adjusted reward for the episode
        total_reward += adjusted_reward

        # Print the current step, original reward, and adjusted reward
        print(f"Episode {episode+1}, Step: {step}, Reward: {reward}, Adjusted Reward: {adjusted_reward}")

        # Check if the episode is done (if the pole fell or the max steps were reached)
        if done or truncated:
            print(f"Episode {episode+1} finished after {step+1} steps with total reward: {total_reward}")
            break

    # Apply epsilon decay after each episode
    epsilon = max(min_epsilon, epsilon * decay_rate)

# Close the environment after running all episodes
env.close()