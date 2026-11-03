"""Central settings for the Quad Loco project.

Values marked TODO must be confirmed against the Isaac Lab 3
project template and selected quadruped before running.
"""

# Project
PROJECT_NAME = "Quad Loco"

# Robot — confirm these from the selected Isaac Lab sample
ROBOT_NAME = "TODO"
NUM_JOINTS = None

# DreamWaQ-inspired policy inputs
OBSERVATION_HISTORY = 5  # Verify exact history convention in implementation
CONTEXT_DIM = None       # TODO: choose/verify from network design

# Simulation — set from the generated task configuration
NUM_ENVS = 64            # Small debugging starting point
EPISODE_LENGTH_SECONDS = None
SIMULATION_DT = None

# PPO — reference values from the DreamWaQ paper;
# verify compatibility with the Isaac Lab RSL-RL version.
PPO_LEARNING_RATE = 1e-3
PPO_DISCOUNT_FACTOR = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2
PPO_ITERATIONS = 1000