# testing.py
import time
import torch
import numpy as np
from environment import create_env  # Ensure you have an environment creation function
from agent import QNetwork, select_action
from utils import add_noise_to_state  # Assuming you have a noise function in utils
from torch.utils.tensorboard import SummaryWriter

# Set the device to CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters (directly in this script instead of a config.py)
TEST_EPISODES = 50
MAX_STEPS = 500
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.4]

# TensorBoard writer setup
general_writer = SummaryWriter(log_dir="logs")  # Change the path as needed

# Initialize the environment and the Q-network
env = create_env()
q_network = QNetwork()
q_network.load_state_dict(torch.load("dqn_acrobot_model.pth", map_location=device))  # Load the trained model
q_network.to(device)
q_network.eval()  # Set the model to evaluation mode

# Initialize results dictionary
testing_results = {
    "noise_level": [],
    "average_reward": [],
    "std_reward": [],
    "average_duration": []
}

# Timing for testing phase
testing_start_time = time.time()

# Improved Testing loop with detailed print statements for each episode
for noise_level in NOISE_LEVELS:
    print(f"\nTesting with Noise Level: {noise_level}")
    eval_rewards = []
    episode_durations = []

    for episode in range(TEST_EPISODES):
        episode_start_time = time.time()  # Start time for each test episode
        state, _ = env.reset()  # Initialize the environment and get the starting state
        state = torch.tensor(state, dtype=torch.float32).to(device)  # Move state to the correct device
        total_reward = 0  # Initialize the total reward counter
        done = False  # Boolean to check if the episode is complete
        step_count = 0  # Step counter to limit maximum steps per episode

        while not done and step_count < MAX_STEPS:
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
        print(f"Test {episode + 1}/{TEST_EPISODES}: Noise Level {noise_level}, Total Reward: {total_reward:.2f}, "
              f"Duration: {episode_duration:.2f} seconds, Steps: {step_count}")

    # Calculate statistics for this noise level
    avg_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    avg_duration = np.mean(episode_durations)

    # Log the results to TensorBoard
    general_writer.add_scalar(f"Testing/Noise_{noise_level}/Average_Reward", avg_reward)
    general_writer.add_scalar(f"Testing/Noise_{noise_level}/Std_Reward", std_reward)
    general_writer.add_scalar(f"Testing/Noise_{noise_level}/Avg_Duration", avg_duration)

    # Print summary for current noise level
    print(f"\nSummary for Noise Level {noise_level}:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Standard Deviation of Reward: {std_reward:.2f}")
    print(f"Average Episode Duration: {avg_duration:.2f} seconds")

    # Append results to the dictionary for summary
    testing_results["noise_level"].append(noise_level)
    testing_results["average_reward"].append(avg_reward)
    testing_results["std_reward"].append(std_reward)
    testing_results["average_duration"].append(avg_duration)

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

# Close the environment after testing
env.close()
