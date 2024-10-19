# !pip
# install
# gymnasium
# !pip
# install
# numpy
# !pip
# install
# matplotlib
# !pip
# install
# stable - baselines3

import gymnasium as gym
import time
from stable_baselines3 import SAC
import numpy as np

# Create the MountainCarContinuous environment using Gymnasium
env = gym.make('MountainCarContinuous-v0')

# Initialize the SAC model
model = SAC("MlpPolicy", env, verbose=1, learning_rate=0.001, gamma=0.99, buffer_size=1000000, batch_size=256)

# Train the SAC agent
model.learn(total_timesteps=100000)

# Save the trained model
model.save("sac_mountaincar")

# Task switching parameters
switch_threshold = 200  # Switch tasks after 200 steps (example)


# Function to modify rewards based on task switching
def adjust_reward(step, original_reward, position, velocity):
    if step < switch_threshold:
        # Task 1: Reward is based on normal environment conditions
        adjusted_reward = original_reward
    else:
        # Task 2: Adjust reward differently based on conditions
        if position > 0.5:
            adjusted_reward = 2 * original_reward
        else:
            adjusted_reward = original_reward - 1

        if abs(velocity) > 0.03:
            adjusted_reward -= 1

    return adjusted_reward


# To test the agent after training with task switching
obs, info = env.reset()

# Inject noise into the initial state observation (optional, if needed)
noise_factor = 0.1
obs = obs + noise_factor * np.random.randn(*np.array(obs).shape)

for step in range(1000):
    action, _states = model.predict(obs, deterministic=True)  # Choose action based on the trained model
    obs, reward, done, truncated, info = env.step(action)
    env.render()

    # Adjust reward based on task switching logic
    position, velocity = obs[0], obs[1]
    adjusted_reward = adjust_reward(step, reward, position, velocity)

    time.sleep(0.05)  # Slow down rendering to observe behavior
    print(f"Step: {step}, Action: {action}, Reward: {reward}, Adjusted Reward: {adjusted_reward}")

    if done or truncated:
        break

# Close the environment after running all episodes
env.close()
