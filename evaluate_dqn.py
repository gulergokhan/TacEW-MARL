import os
import random
import numpy as np
import torch

from environment.tactical_env import TacticalEnv
from agent.dqn_agent import DQNAgent


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/tactical_dqn_goal_aware.pth"

NUM_EPISODES = 100
MAX_STEPS = 100

# IMPORTANT:
# The current tactical DQN was trained with the observation
# returned directly by TacticalEnv.reset().
#
# Current observation dimension:
# 14 base features
# + 45 terrain features
# + 42 radar features
# = 101
STATE_SIZE = 101

ACTION_SIZE = 7

SEED = 42


# ============================================================
# SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# OBSERVATION HELPER
# ============================================================

def get_observation(reset_result):
    """
    TacticalEnv currently returns the observation directly.

    This helper also supports Gymnasium-style:
        (observation, info)

    so the evaluator remains robust.
    """

    if isinstance(reset_result, tuple):

        if len(reset_result) >= 1:
            observation = reset_result[0]
        else:
            raise RuntimeError(
                "TacticalEnv.reset() returned an empty tuple."
            )

    else:

        observation = reset_result

    observation = np.asarray(
        observation,
        dtype=np.float32
    )

    return observation


def get_step_result(step_result):
    """
    Supports both:

        obs, reward, done, info

    and Gymnasium:

        obs, reward, terminated, truncated, info
    """

    if len(step_result) == 4:

        next_obs, reward, done, info = step_result

        return (
            next_obs,
            float(reward),
            bool(done),
            info
        )

    if len(step_result) == 5:

        next_obs, reward, terminated, truncated, info = step_result

        done = bool(
            terminated or truncated
        )

        return (
            next_obs,
            float(reward),
            done,
            info
        )

    raise RuntimeError(
        "Unexpected TacticalEnv.step() return format: "
        f"{len(step_result)} values."
    )


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def extract_state_dict(checkpoint):

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            return checkpoint["state_dict"]

        if "model_state_dict" in checkpoint:

            return checkpoint["model_state_dict"]

        if any(
            key.startswith("network.")
            for key in checkpoint.keys()
        ):

            return checkpoint

    raise RuntimeError(
        "Could not find DQN state_dict in checkpoint."
    )


def get_episode_training_info(checkpoint):

    if not isinstance(checkpoint, dict):

        return None, None

    episode = checkpoint.get(
        "episode"
    )

    reward = checkpoint.get(
        "best_reward"
    )

    return episode, reward


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 60)
    print("TALON - DQN EVALUATION")
    print("=" * 60)

    print(
        f"Model             : "
        f"{os.path.abspath(MODEL_PATH)}"
    )

    print(
        f"Episodes          : "
        f"{NUM_EPISODES}"
    )

    print(
        "Exploration       : OFF"
    )

    print(
        "Observation       : Current 101D"
    )

    print(
        f"Action space      : "
        f"{ACTION_SIZE}"
    )

    print(
        "Device            : cpu"
    )

    print("=" * 60)

    # ========================================================
    # ENVIRONMENT
    # ========================================================

    env = TacticalEnv(
        width=10,
        height=10,
        latitude=39.9334,
        longitude=32.8597,
    )

    # ========================================================
    # INITIAL OBSERVATION
    # ========================================================

    reset_result = env.reset()

    initial_state = get_observation(
        reset_result
    )

    print(
        f"Observation shape : "
        f"{initial_state.shape}"
    )

    if initial_state.shape != (STATE_SIZE,):

        raise RuntimeError(
            "DQN observation dimension mismatch. "
            f"Expected ({STATE_SIZE},), "
            f"got {initial_state.shape}."
        )

    print(
        "Observation check : PASS"
    )

    # ========================================================
    # DQN
    # ========================================================

    agent = DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        learning_rate=0.001,
        gamma=0.99,
        buffer_size=10000,
        batch_size=64,
        epsilon=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        target_update_freq=500,
    )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    agent.model.load_state_dict(
        state_dict
    )

    agent.model.eval()

    # No exploration
    agent.epsilon = 0.0

    training_episode, training_reward = (
        get_episode_training_info(
            checkpoint
        )
    )

    print()

    if training_episode is not None:

        print(
            f"Training episode  : "
            f"{training_episode}"
        )

    if training_reward is not None:

        print(
            f"Training reward   : "
            f"{training_reward}"
        )

    print("-" * 60)

    # ========================================================
    # METRICS
    # ========================================================

    rewards = []
    scout_rewards = []
    hunter_rewards = []
    episode_steps = []

    successful_episodes = 0
    radar_failures = 0
    mission_successes = 0

    # ========================================================
    # ACTION COUNTS
    # ========================================================

    action_counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
    }

    action_names = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT",
        4: "STAY",
        5: "JAM_SUPPRESS",
        6: "JAM_DECEIVE",
    }

    episode_action_counts = []

    # ========================================================
    # EVALUATION
    # ========================================================

    for episode in range(
        1,
        NUM_EPISODES + 1
    ):

        # ----------------------------------------------------
        # RESET
        # ----------------------------------------------------

        reset_result = env.reset()

        state = get_observation(
            reset_result
        )

        if state.shape != (STATE_SIZE,):

            raise RuntimeError(
                f"Episode {episode}: "
                f"invalid observation shape "
                f"{state.shape}; "
                f"expected ({STATE_SIZE},)."
            )

        total_reward = 0.0
        total_scout_reward = 0.0
        total_hunter_reward = 0.0

        steps = 0

        episode_actions = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
        }

        episode_radar_failure = False
        episode_mission_success = False

        # ====================================================
        # EPISODE LOOP
        # ====================================================

        for step in range(
            1,
            MAX_STEPS + 1
        ):

            # ------------------------------------------------
            # GREEDY ACTION
            # ------------------------------------------------

            action = agent.select_action(
                state,
                training=False
            )

            action = int(action)

            if action not in action_counts:

                raise RuntimeError(
                    f"Invalid action returned by DQN: "
                    f"{action}"
                )

            action_counts[action] += 1
            episode_actions[action] += 1

            # ------------------------------------------------
            # ENVIRONMENT STEP
            # ------------------------------------------------

            step_result = env.step(
                action
            )

            (
                next_obs,
                reward,
                done,
                info
            ) = get_step_result(
                step_result
            )

            next_state = np.asarray(
                next_obs,
                dtype=np.float32
            )

            if next_state.shape != (STATE_SIZE,):

                raise RuntimeError(
                    f"Episode {episode}, step {step}: "
                    f"invalid next observation shape "
                    f"{next_state.shape}; "
                    f"expected ({STATE_SIZE},)."
                )

            # ------------------------------------------------
            # REWARD
            # ------------------------------------------------

            total_reward += reward

            scout_reward = float(
                info.get(
                    "reward_scout",
                    reward
                )
            )

            hunter_reward = float(
                info.get(
                    "reward_hunter",
                    reward
                )
            )

            total_scout_reward += (
                scout_reward
            )

            total_hunter_reward += (
                hunter_reward
            )

            steps = step

            # ------------------------------------------------
            # EVENTS
            # ------------------------------------------------

            if info.get(
                "radar_failure",
                False
            ):

                episode_radar_failure = True

            if info.get(
                "mission_success",
                False
            ):

                episode_mission_success = True

            # ------------------------------------------------
            # FIRST EPISODE DIAGNOSTIC
            # ------------------------------------------------

            if episode == 1:

                scout = getattr(
                    env,
                    "scout",
                    None
                )

                hunter = getattr(
                    env,
                    "hunter",
                    None
                )

                scout_position = getattr(
                    scout,
                    "position",
                    None
                )

                hunter_position = getattr(
                    hunter,
                    "position",
                    None
                )

                print(
                    f"Step {step:3d} | "
                    f"Action={action_names[action]:12s} | "
                    f"Reward={reward:7.2f}"
                )

                if scout_position is not None:

                    print(
                        f"           Scout  = "
                        f"{scout_position}"
                    )

                if hunter_position is not None:

                    print(
                        f"           Hunter = "
                        f"{hunter_position}"
                    )

            # ------------------------------------------------
            # NEXT STATE
            # ------------------------------------------------

            state = next_state

            if done:

                break

        # ====================================================
        # EPISODE RESULT
        # ====================================================

        episode_success = (
            episode_mission_success
        )

        if episode_success:

            successful_episodes += 1

        if episode_radar_failure:

            radar_failures += 1

        if episode_mission_success:

            mission_successes += 1

        rewards.append(
            total_reward
        )

        scout_rewards.append(
            total_scout_reward
        )

        hunter_rewards.append(
            total_hunter_reward
        )

        episode_steps.append(
            steps
        )

        episode_action_counts.append(
            episode_actions
        )

        print(
            f"Episode {episode:3d} | "
            f"Reward {total_reward:8.2f} | "
            f"Steps {steps:3d} | "
            f"Radar Failure "
            f"{'YES' if episode_radar_failure else 'NO ':3s} | "
            f"Success "
            f"{'YES' if episode_success else 'NO'}"
        )

    # ========================================================
    # NUMPY METRICS
    # ========================================================

    rewards_np = np.asarray(
        rewards,
        dtype=np.float32
    )

    scout_rewards_np = np.asarray(
        scout_rewards,
        dtype=np.float32
    )

    hunter_rewards_np = np.asarray(
        hunter_rewards,
        dtype=np.float32
    )

    steps_np = np.asarray(
        episode_steps,
        dtype=np.float32
    )

    average_reward = float(
        np.mean(rewards_np)
    )

    average_scout_reward = float(
        np.mean(scout_rewards_np)
    )

    average_hunter_reward = float(
        np.mean(hunter_rewards_np)
    )

    average_steps = float(
        np.mean(steps_np)
    )

    success_rate = (
        100.0
        * successful_episodes
        / NUM_EPISODES
    )

    radar_failure_rate = (
        100.0
        * radar_failures
        / NUM_EPISODES
    )

    mission_success_rate = (
        100.0
        * mission_successes
        / NUM_EPISODES
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print()
    print("=" * 60)
    print("DQN EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"Average Reward     : "
        f"{average_reward:.2f}"
    )

    print(
        f"Success Rate       : "
        f"{success_rate:.2f}%"
    )

    print(
        f"Average Steps      : "
        f"{average_steps:.2f}"
    )

    print(
        f"Scout Reward       : "
        f"{average_scout_reward:.2f}"
    )

    print(
        f"Hunter Reward      : "
        f"{average_hunter_reward:.2f}"
    )

    print(
        f"Radar Failure      : "
        f"{radar_failure_rate:.2f}%"
    )

    print(
        f"Mission Success    : "
        f"{mission_success_rate:.2f}%"
    )

    # ========================================================
    # ADDITIONAL DIAGNOSTICS
    # ========================================================

    print()
    print("=" * 60)
    print("ADDITIONAL DIAGNOSTICS")
    print("=" * 60)

    print(
        f"Successful episodes : "
        f"{successful_episodes}/{NUM_EPISODES}"
    )

    print(
        f"Radar failures      : "
        f"{radar_failures}/{NUM_EPISODES}"
    )

    print(
        f"Mission successes   : "
        f"{mission_successes}/{NUM_EPISODES}"
    )

    print(
        f"Min reward          : "
        f"{np.min(rewards_np):.2f}"
    )

    print(
        f"Max reward          : "
        f"{np.max(rewards_np):.2f}"
    )

    print(
        f"Reward std          : "
        f"{np.std(rewards_np):.2f}"
    )

    # ========================================================
    # ACTION DISTRIBUTION
    # ========================================================

    print()
    print("=" * 60)
    print("ACTION DISTRIBUTION")
    print("=" * 60)

    total_actions = sum(
        action_counts.values()
    )

    for action_id in range(
        ACTION_SIZE
    ):

        count = action_counts[
            action_id
        ]

        percentage = (
            100.0
            * count
            / total_actions
            if total_actions > 0
            else 0.0
        )

        print(
            f"{action_id} "
            f"{action_names[action_id]:12s} : "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )

    print(
        f"Total actions      : "
        f"{total_actions}"
    )

    # ========================================================
    # MOST USED ACTION
    # ========================================================

    most_used_action = max(
        action_counts,
        key=action_counts.get
    )

    print()

    print(
        f"Most used action   : "
        f"{most_used_action} "
        f"({action_names[most_used_action]})"
    )

    # ========================================================
    # PER-EPISODE ACTION STATISTICS
    # ========================================================

    print()
    print("=" * 60)
    print("PER-EPISODE ACTION STATISTICS")
    print("=" * 60)

    for action_id in range(
        ACTION_SIZE
    ):

        values = [
            episode_counts[action_id]
            for episode_counts
            in episode_action_counts
        ]

        print(
            f"{action_names[action_id]:12s} | "
            f"avg {np.mean(values):7.2f} | "
            f"min {np.min(values):4d} | "
            f"max {np.max(values):4d}"
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()