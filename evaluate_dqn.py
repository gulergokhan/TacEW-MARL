import os
import random
import numpy as np
import torch

from environment.tactical_env import TacticalEnv
from agent.dqn_agent import DQNAgent


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/tactical_dqn.pth"

NUM_EPISODES = 100
MAX_STEPS = 100

STATE_SIZE = 84
ACTION_SIZE = 7

SEED = 42


# ============================================================
# LEGACY 84D OBSERVATION ENCODER
# ============================================================

class LegacyDQNObservationEncoder:
    """
    Recreates the original 84-dimensional DQN observation.

    12 base features
    + 45 terrain features
    + 27 radar features
    = 84
    """

    MAX_RADARS = 3

    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height

    # --------------------------------------------------------
    # AIRCRAFT HELPERS
    # --------------------------------------------------------

    @staticmethod
    def _get_xy(obj):

        if obj is None:
            return 0.0, 0.0

        position = getattr(obj, "position", None)

        if position is None and isinstance(obj, dict):
            position = obj.get("position")

        if position is None:
            x = getattr(obj, "x", 0.0)
            y = getattr(obj, "y", 0.0)

            return float(x), float(y)

        if isinstance(position, dict):
            return (
                float(position.get("x", 0.0)),
                float(position.get("y", 0.0)),
            )

        if hasattr(position, "x") and hasattr(position, "y"):
            return (
                float(position.x),
                float(position.y),
            )

        if isinstance(
            position,
            (tuple, list, np.ndarray)
        ):
            return (
                float(position[0]),
                float(position[1]),
            )

        return 0.0, 0.0

    @staticmethod
    def _get_fuel(obj):

        if obj is None:
            return 0.0

        if isinstance(obj, dict):
            return float(
                obj.get("fuel", 0.0)
            )

        return float(
            getattr(obj, "fuel", 0.0)
        )

    @staticmethod
    def _normalize_position(value, maximum):

        if maximum <= 0:
            return 0.0

        return float(value) / float(maximum)

    @staticmethod
    def _normalize_fuel(fuel):

        return float(fuel) / 100.0

    # --------------------------------------------------------
    # TERRAIN
    # --------------------------------------------------------

    def _terrain_one_hot(self, terrain_type):

        result = [0.0] * 5

        try:
            index = int(terrain_type)
        except Exception:
            index = 0

        if 0 <= index < 5:
            result[index] = 1.0

        return result

    def _get_terrain_type(
        self,
        terrain,
        x,
        y
    ):

        if terrain is None:
            return 0

        if hasattr(terrain, "get_terrain"):

            try:
                return terrain.get_terrain(
                    x,
                    y
                )
            except Exception:
                pass

        if hasattr(terrain, "get_cell"):

            try:

                cell = terrain.get_cell(
                    x,
                    y
                )

                if hasattr(
                    cell,
                    "terrain_type"
                ):
                    return cell.terrain_type

                return cell

            except Exception:
                pass

        grid = getattr(
            terrain,
            "grid",
            None
        )

        if grid is not None:

            try:

                cell = grid[int(y)][int(x)]

                if hasattr(
                    cell,
                    "terrain_type"
                ):
                    return cell.terrain_type

                return cell

            except Exception:
                pass

        return 0

    # --------------------------------------------------------
    # RADAR HELPERS
    # --------------------------------------------------------

    def _radar_state_index(
        self,
        radar_state
    ):

        if radar_state is None:
            return 0

        value = getattr(
            radar_state,
            "value",
            radar_state
        )

        value = str(value).upper()

        mapping = {
            "SAFE": 0,
            "TRACK": 1,
            "LOCK": 2,
            "LETHAL": 3,
        }

        return mapping.get(
            value,
            0
        )

    def _get_radar_position(
        self,
        radar
    ):

        return self._get_xy(radar)

    def _get_radar_detection_range(
        self,
        radar
    ):

        if radar is None:
            return 0.0

        if isinstance(radar, dict):

            return float(
                radar.get(
                    "detection_range",
                    0.0
                )
            )

        return float(
            getattr(
                radar,
                "detection_range",
                getattr(
                    radar,
                    "range",
                    0.0
                )
            )
        )

    def _get_radar_active(
        self,
        radar
    ):

        if radar is None:
            return 0.0

        if isinstance(radar, dict):

            return (
                1.0
                if radar.get(
                    "active",
                    False
                )
                else 0.0
            )

        return (
            1.0
            if getattr(
                radar,
                "active",
                False
            )
            else 0.0
        )

    def _get_radar_state(
        self,
        radar,
        agent_name="scout"
    ):

        if radar is None:
            return "SAFE"

        if isinstance(radar, dict):

            if agent_name == "scout":

                return radar.get(
                    "scout_state",
                    radar.get(
                        "state",
                        "SAFE"
                    )
                )

            return radar.get(
                "state",
                "SAFE"
            )

        if agent_name == "scout":

            return getattr(
                radar,
                "scout_state",
                getattr(
                    radar,
                    "state",
                    "SAFE"
                )
            )

        return getattr(
            radar,
            "state",
            "SAFE"
        )

    @staticmethod
    def _distance(
        x1,
        y1,
        x2,
        y2
    ):

        return float(
            np.sqrt(
                (x1 - x2) ** 2
                +
                (y1 - y2) ** 2
            )
        )

    # ========================================================
    # ENCODE
    # ========================================================

    def encode(self, env):

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

        terrain = getattr(
            env,
            "terrain",
            None
        )

        radar_system = getattr(
            env,
            "radar_system",
            None
        )

        obs = []

        # ====================================================
        # 1. AIRCRAFT FEATURES
        # ====================================================

        scout_x, scout_y = self._get_xy(
            scout
        )

        hunter_x, hunter_y = self._get_xy(
            hunter
        )

        scout_fuel = self._get_fuel(
            scout
        )

        hunter_fuel = self._get_fuel(
            hunter
        )

        # Scout
        obs.append(
            self._normalize_position(
                scout_x,
                self.width - 1
            )
        )

        obs.append(
            self._normalize_position(
                scout_y,
                self.height - 1
            )
        )

        obs.append(
            self._normalize_fuel(
                scout_fuel
            )
        )

        # Hunter
        obs.append(
            self._normalize_position(
                hunter_x,
                self.width - 1
            )
        )

        obs.append(
            self._normalize_position(
                hunter_y,
                self.height - 1
            )
        )

        obs.append(
            self._normalize_fuel(
                hunter_fuel
            )
        )

        # ====================================================
        # 2. WEATHER
        # ====================================================

        weather = getattr(
            env,
            "weather",
            None
        )

        weather_values = []

        if weather is None:

            weather_values = [
                0.0
            ] * 6

        else:

            possible_names = [
                "temperature",
                "wind_speed",
                "wind_direction",
                "visibility",
                "precipitation",
                "cloud_cover",
            ]

            for name in possible_names:

                if isinstance(
                    weather,
                    dict
                ):

                    value = weather.get(
                        name,
                        0.0
                    )

                else:

                    value = getattr(
                        weather,
                        name,
                        0.0
                    )

                try:
                    weather_values.append(
                        float(value)
                    )
                except Exception:
                    weather_values.append(
                        0.0
                    )

            while len(
                weather_values
            ) < 6:

                weather_values.append(
                    0.0
                )

            weather_values = (
                weather_values[:6]
            )

        obs.extend(
            weather_values
        )

        # ====================================================
        # 3. LOCAL TERRAIN 3x3
        # ====================================================

        center_x = int(
            round(scout_x)
        )

        center_y = int(
            round(scout_y)
        )

        for dy in (-1, 0, 1):

            for dx in (-1, 0, 1):

                x = center_x + dx
                y = center_y + dy

                if (
                    x < 0
                    or x >= self.width
                    or y < 0
                    or y >= self.height
                ):

                    terrain_type = 0

                else:

                    terrain_type = (
                        self._get_terrain_type(
                            terrain,
                            x,
                            y
                        )
                    )

                obs.extend(
                    self._terrain_one_hot(
                        terrain_type
                    )
                )

        # ====================================================
        # 4. RADARS
        #
        # IMPORTANT:
        # Original DQN has ONLY 9 radar features:
        #
        # X                1
        # Y                1
        # Detection range  1
        # Active           1
        # Radar state      4
        # Scout distance   1
        #
        # TOTAL            9
        #
        # 3 radars = 27
        # ====================================================

        radars = []

        if radar_system is not None:

            if hasattr(
                radar_system,
                "radars"
            ):

                radars = (
                    radar_system.radars
                )

            elif isinstance(
                radar_system,
                dict
            ):

                radars = radar_system.get(
                    "radars",
                    []
                )

        if radars is None:
            radars = []

        radars = list(radars)

        for i in range(
            self.MAX_RADARS
        ):

            if i < len(radars):

                radar = radars[i]

                radar_x, radar_y = (
                    self._get_radar_position(
                        radar
                    )
                )

                detection_range = (
                    self._get_radar_detection_range(
                        radar
                    )
                )

                active = (
                    self._get_radar_active(
                        radar
                    )
                )

                # Radar X
                obs.append(
                    self._normalize_position(
                        radar_x,
                        self.width - 1
                    )
                )

                # Radar Y
                obs.append(
                    self._normalize_position(
                        radar_y,
                        self.height - 1
                    )
                )

                # Detection range
                obs.append(
                    detection_range / 10.0
                )

                # Active
                obs.append(
                    active
                )

                # Scout radar state
                scout_state = (
                    self._get_radar_state(
                        radar,
                        "scout"
                    )
                )

                scout_state_index = (
                    self._radar_state_index(
                        scout_state
                    )
                )

                scout_one_hot = [
                    0.0
                ] * 4

                scout_one_hot[
                    scout_state_index
                ] = 1.0

                obs.extend(
                    scout_one_hot
                )

                # Scout distance
                scout_distance = (
                    self._distance(
                        scout_x,
                        scout_y,
                        radar_x,
                        radar_y
                    )
                )

                obs.append(
                    scout_distance / 10.0
                )

            else:

                # Missing radar padding
                obs.extend(
                    [0.0] * 9
                )

        # ====================================================
        # FINAL 84D CHECK
        # ====================================================

        result = np.asarray(
            obs,
            dtype=np.float32
        )

        if result.shape != (84,):

            raise RuntimeError(
                "Legacy DQN observation "
                f"mismatch: expected (84,), "
                f"got {result.shape}"
            )

        return result


# ============================================================
# HELPERS
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_state_dict(
    checkpoint
):

    if isinstance(
        checkpoint,
        dict
    ):

        if "state_dict" in checkpoint:

            return checkpoint[
                "state_dict"
            ]

        if "model_state_dict" in checkpoint:

            return checkpoint[
                "model_state_dict"
            ]

        if any(
            key.startswith(
                "network."
            )
            for key in checkpoint.keys()
        ):

            return checkpoint

    raise RuntimeError(
        "Could not find DQN "
        "state_dict in checkpoint."
    )


def get_episode_training_info(
    checkpoint
):

    if not isinstance(
        checkpoint,
        dict
    ):

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
        "Observation       : Legacy 84D"
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

    encoder = LegacyDQNObservationEncoder(
        width=10,
        height=10,
    )

    # ========================================================
    # INITIAL OBSERVATION
    # ========================================================

    env.reset()

    initial_state = encoder.encode(
        env
    )

    print(
        f"Observation shape : "
        f"{initial_state.shape}"
    )

    if initial_state.shape != (84,):

        raise RuntimeError(
            "DQN requires an "
            "84-dimensional observation."
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

    # Action counts
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

        env.reset()

        state = encoder.encode(
            env
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

            action = agent.select_action(
                state,
                training=False
            )

            action = int(action)

            action_counts[
                action
            ] += 1

            episode_actions[
                action
            ] += 1

            # ------------------------------------------------
            # ENV STEP
            # ------------------------------------------------

            next_obs, reward, done, info = (
                env.step(action)
            )

            total_reward += float(
                reward
            )

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

                print(
                    f"Step {step:3d} | "
                    f"Action: {action} | "
                    f"Reward: {reward:7.2f} | "
                    f"Scout R: {scout_reward:7.2f} | "
                    f"Hunter R: {hunter_reward:7.2f}"
                )

            # ------------------------------------------------
            # NEXT STATE
            # ------------------------------------------------

            state = encoder.encode(
                env
            )

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
            episode_counts[
                action_id
            ]
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