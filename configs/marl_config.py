# ==================================================
# MARL / HAPPO CONFIGURATION
# ==================================================

# Observation / action dimensions
SCOUT_OBS_DIM = 87
HUNTER_OBS_DIM = 87
GLOBAL_STATE_DIM = 101
ACTION_DIM = 7

# Network
HIDDEN_DIM = 128

# PPO / HAPPO
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 3e-4

GAMMA = 0.99
GAE_LAMBDA = 0.95

CLIP_EPSILON = 0.2

ENTROPY_COEF = 0.01
VALUE_COEF = 0.5

PPO_EPOCHS = 10
MINIBATCH_SIZE = 64

# Training
ROLLOUT_LENGTH = 256
NUM_EPISODES = 1000

# Device
DEVICE = "cpu"
