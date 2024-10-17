import gymnasium as gym
import numpy as np

# Create the CartPole environment using Gymnasium
env = gym.make('CartPole-v1')

# Parameters
switch_threshold = 10  # Step threshold to switch reward structures
noise_factor = 0.1  # Noise level for observation noise injection
num_episodes = 100  # Increased number of episodes to run

# Initializing the Q-table: (number of states, number of actions)
# Cartpole doesn't have a discrete observation space, so this is recommended discreting metrics
# Number of discrete states for each dimension (position, velocity, angle, angular velocity)
n_bins = (6,12)
q_table = np.zeros([env.observation_space.shape[0], env.action_space.shape.n])
# env.observation_space = Searches the entire search space of the enviorment
# In cartpole, the state consists of four continuous values: the cart’s position,
# cart’s velocity, pole angle, and pole's angular velocity. So it will return 4
# env.action_space.n = Actions agent can take, so 2 (left & right)

# Making continous values into discrete
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
    new_state = [int(round((bin[i] - 1) * ratios[i])) for i in range(len(state))]

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
    noise = np.random.randn(*state.shape) * noise_factor
    return state + noise  # Add noise to the state observations

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
    best_next_action = np.argmax(q_table[next_state])
    q_table[state][action] = q_table[state][action] + alpha * (
        reward + gamma * q_table[next_state][best_next_action] - q_table[state][action]
    )

# Running multiple episodes
for episode in range(num_episodes):
    # Reset the environment to start a new episode
    state, info = env.reset()

    total_reward = 0  # Track total reward in each episode
    adjusted_reward = 0  # Initialize the adjusted reward

    for step in range(5000000):  # Increased to 500 steps per episode
        # Inject noise into the observations (optional)
        noisy_state = inject_noise(state, noise_factor)

        # Take a random action (you can replace this with an RL agent's action)
        action = env.action_space.sample()

        # Perform the action in the environment
        state, reward, done, truncated, info = env.step(action)

        # Task switching based on the step count (rewards change after switch_threshold steps)
        if step < switch_threshold:
            # Task 1: Use the standard reward (no changes)
            adjusted_reward = reward
        else:
            # Task 2: Modify the reward based on the pole's angle (state[2] is the pole angle)
            pole_angle = state[2]  # Get the pole angle from the state
            if abs(pole_angle) < 0.1:
                # Give a higher reward if the pole is near upright
                adjusted_reward = 2 * reward
            else:
                # Penalize if the pole is at a steep angle (about to fall)
                adjusted_reward = reward - 1

        # Accumulate the adjusted reward for the episode
        total_reward += adjusted_reward

        # Print the current step, original reward, and adjusted reward
        print(f"Episode {episode+1}, Step: {step}, Reward: {reward}, Adjusted Reward: {adjusted_reward}")

        # Check if the episode is done (if the pole fell or the max steps were reached)
        if done or truncated:
            print(f"Episode {episode+1} finished after {step+1} steps with total reward: {total_reward}")
            break

# Close the environment after running all episodes
env.close()
