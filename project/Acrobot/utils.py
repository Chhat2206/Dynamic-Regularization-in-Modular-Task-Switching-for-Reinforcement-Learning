import numpy as np

# Function to add noise to a state
def add_noise_to_state(state, noise_level):
    if noise_level > 0:
        noise = np.random.normal(loc=0.0, scale=noise_level, size=state.shape)
        noisy_state = state + noise
    else:
        noisy_state = state
    return noisy_state
