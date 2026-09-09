# ==================================================
# MARL / HAPPO CONFIGURATION
# ==================================================

# Observation / action dimensions
SCOUT_OBS_DIM = 87
HUNTER_OBS_DIM = 87
GLOBAL_STATE_DIM = 101
SCOUT_ACTION_DIM = 7
HUNTER_ACTION_DIM = 5
TRAINING_GRID_SIZE = 10

# Network
HIDDEN_DIM = 128

# PPO / HAPPO
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 3e-4

GAMMA = 0.99
GAE_LAMBDA = 0.95

CLIP_EPSILON = 0.2

ENTROPY_COEF = 0.003
ENTROPY_COEF_END = 0.0005
VALUE_COEF = 0.5

PPO_EPOCHS = 5
MINIBATCH_SIZE = 64

# Expert warm start
EXPERT_PRETRAINING_ENABLED = True
EXPERT_RADAR_LAYOUT_COUNT = 12
EXPERT_INCLUDE_NO_RADAR_LAYOUT = True
EXPERT_DEMONSTRATION_REPEATS = 3
EXPERT_DEFAULT_LAYOUT_REPEAT_MULTIPLIER = 1
EXPERT_PRETRAINING_EPOCHS = 600
EXPERT_PRETRAINING_LR = 1e-3
EXPERT_PRETRAINING_BATCH_SIZE = 64

# Guided scratch training starts from random weights, teaches the two
# role policies a balanced action vocabulary, and then continues with
# on-policy HAPPO updates. Periodic one-epoch refreshes prevent the
# target-conditioned policy from collapsing to a single direction.
SCRATCH_GUIDED_WARMUP_ENABLED = True
SCRATCH_GUIDED_MAX_EPOCHS = 600
SCRATCH_GUIDED_BLOCK_EPOCHS = 100
SCRATCH_GUIDANCE_UPDATE_INTERVAL = 4
SCRATCH_GUIDANCE_REFRESH_EPOCHS = 1

# Rotate through every synthetic radar family during the full curriculum.
# Hand-authored holdout maps remain unseen by training.
HAPPO_TRAINING_RADAR_LAYOUT_INDICES = tuple(
    range(EXPERT_RADAR_LAYOUT_COUNT)
)
RADAR_VALIDATION_LAYOUT_INDICES = (1, 3, 5, 7, 9, 11)
RADAR_VALIDATION_EPISODES = 3

# Training
ROLLOUT_LENGTH = 256
NUM_EPISODES = 5000
PRINT_INTERVAL = 50
SEED = 42
BEST_MODEL_WINDOW = 100
VALIDATION_INTERVAL = 100
VALIDATION_EPISODES_PER_CASE = 5
EARLY_STOP_WORST_SUCCESS = 100.0
EARLY_STOP_PATIENCE = 3

CURRICULUM_STAGE_1_END = 500
CURRICULUM_STAGE_2_END = 1500
BEST_MODEL_START_EPISODE = VALIDATION_INTERVAL

TRAINING_START_FORMATIONS = [
    ((7, 7), (8, 8)),
    ((7, 8), (8, 8)),
    ((6, 7), (8, 8)),
    ((8, 5), (8, 6)),
    ((8, 7), (9, 8)),
    ((8, 8), (9, 9)),
    ((4, 7), (5, 8)),
]

TRAINING_STRIKE_POINTS = [
    (1, 8),
    (8, 0),
    (1, 1),
]

# Device
DEVICE = "cpu"
