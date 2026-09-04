import random
import numpy as np
import torch

from environment.tactical_env import TacticalEnv
from agent.dqn_agent import DQNAgent


MODEL_PATH = "models/tactical_dqn.pth"

STATE_SIZE = 84
ACTION_SIZE = 7

MAX_STEPS = 100
SEED = 42


# ============================================================
# LEGACY 84D ENCODER
# ============================================================

class LegacyDQNObservationEncoder:

    MAX_RADARS = 3

    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height

    @staticmethod
    def get_xy(obj):

        if obj is None:
            return 0.0, 0.0

        position = getattr(
            obj,
            "position",
            None
        )

        if position is None:
            x = getattr(obj, "x", 0.0)
            y = getattr(obj, "y", 0.0)
            return float(x), float(y)

        if hasattr(position, "x"):
            return (
                float(position.x),
                float(position.y)
            )

        if isinstance(
            position,
            (tuple, list, np.ndarray)
        ):
            return (
                float(position[0]),
                float(position[1])
            )

        if isinstance(position, dict):
            return (
                float(position.get("x", 0)),
                float(position.get("y", 0))
            )

        return 0.0, 0.0

    @staticmethod
    def get_fuel(obj):

        if obj is None:
            return 0.0

        return float(
            getattr(
                obj,
                "fuel",
                0.0
            )
        )

    def terrain_one_hot(self, terrain_type):

        result = [0.0] * 5

        try:
            index = int(terrain_type)
        except Exception:
            index = 0

        if 0 <= index < 5:
            result[index] = 1.0

        return result

    def get_terrain(self, terrain, x, y):

        if terrain is None:
            return 0

        if hasattr(
            terrain,
            "get_terrain"
        ):

            try:
                return terrain.get_terrain(
                    x,
                    y
                )
            except Exception:
                pass

        if hasattr(
            terrain,
            "get_cell"
        ):

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

        return 0

    def radar_state_index(self, state):

        if state is None:
            return 0

        value = getattr(
            state,
            "value",
            state
        )

        value = str(value).upper()

        return {
            "SAFE": 0,
            "TRACK": 1,
            "LOCK": 2,
            "LETHAL": 3,
        }.get(value, 0)

    def get_radar_state(
        self,
        radar
    ):

        state = getattr(
            radar,
            "state",
            "SAFE"
        )

        return state

    def encode(self, env):

        scout = env.scout
        hunter = env.hunter
        terrain = env.terrain
        radar_system = env.radar_system

        scout_x, scout_y = self.get_xy(
            scout
        )

        hunter_x, hunter_y = self.get_xy(
            hunter
        )

        scout_fuel = self.get_fuel(
            scout
        )

        hunter_fuel = self.get_fuel(
            hunter
        )

        obs = []

        # ----------------------------------------------------
        # AIRCRAFT = 6
        # ----------------------------------------------------

        obs.extend([
            scout_x / (self.width - 1),
            scout_y / (self.height - 1),
            scout_fuel / 100.0,

            hunter_x / (self.width - 1),
            hunter_y / (self.height - 1),
            hunter_fuel / 100.0,
        ])

        # ----------------------------------------------------
        # WEATHER = 6
        # ----------------------------------------------------

        weather = getattr(
            env,
            "weather",
            None
        )

        weather_names = [
            "temperature",
            "wind_speed",
            "wind_direction",
            "visibility",
            "precipitation",
            "cloud_cover",
        ]

        for name in weather_names:

            value = getattr(
                weather,
                name,
                0.0
            )

            try:
                obs.append(
                    float(value)
                )
            except Exception:
                obs.append(0.0)

        # ----------------------------------------------------
        # TERRAIN 3x3 = 45
        # ----------------------------------------------------

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
                        self.get_terrain(
                            terrain,
                            x,
                            y
                        )
                    )

                obs.extend(
                    self.terrain_one_hot(
                        terrain_type
                    )
                )

        # ----------------------------------------------------
        # RADARS = 27
        # ----------------------------------------------------

        radars = getattr(
            radar_system,
            "radars",
            []
        )

        for i in range(3):

            if i < len(radars):

                radar = radars[i]

                radar_x, radar_y = (
                    self.get_xy(radar)
                )

                detection_range = float(
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

                active = (
                    1.0
                    if getattr(
                        radar,
                        "active",
                        False
                    )
                    else 0.0
                )

                # X
                obs.append(
                    radar_x / (
                        self.width - 1
                    )
                )

                # Y
                obs.append(
                    radar_y / (
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
                state = self.get_radar_state(
                    radar
                )

                state_index = (
                    self.radar_state_index(
                        state
                    )
                )

                one_hot = [0.0] * 4

                one_hot[
                    state_index
                ] = 1.0

                obs.extend(one_hot)

                # Scout distance
                distance = np.sqrt(
                    (scout_x - radar_x) ** 2
                    +
                    (scout_y - radar_y) ** 2
                )

                obs.append(
                    float(distance) / 10.0
                )

            else:

                obs.extend(
                    [0.0] * 9
                )

        result = np.asarray(
            obs,
            dtype=np.float32
        )

        assert result.shape == (84,), (
            f"Expected 84D, got {result.shape}"
        )

        return result


# ============================================================
# HELPERS
# ============================================================

def print_object(
    name,
    obj
):

    if obj is None:

        print(
            f"{name}: None"
        )

        return

    position = getattr(
        obj,
        "position",
        None
    )

    fuel = getattr(
        obj,
        "fuel",
        None
    )

    print(
        f"{name}: "
        f"position={position}, "
        f"fuel={fuel}"
    )


def print_radar_info(env):

    radar_system = getattr(
        env,
        "radar_system",
        None
    )

    radars = getattr(
        radar_system,
        "radars",
        []
    )

    print(
        f"Radar count: {len(radars)}"
    )

    for i, radar in enumerate(
        radars
    ):

        position = getattr(
            radar,
            "position",
            None
        )

        state = getattr(
            radar,
            "state",
            None
        )

        active = getattr(
            radar,
            "active",
            None
        )

        detection_range = getattr(
            radar,
            "detection_range",
            None
        )

        print(
            f"  Radar {i}: "
            f"position={position} | "
            f"state={state} | "
            f"active={active} | "
            f"range={detection_range}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("TALON - DQN DIAGNOSTIC")
    print("=" * 70)

    print(
        f"Model       : {MODEL_PATH}"
    )

    print(
        "Observation : Legacy 84D"
    )

    print(
        "Exploration : OFF"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # ENV
    # --------------------------------------------------------

    env = TacticalEnv(
        width=10,
        height=10,
        latitude=39.9334,
        longitude=32.8597,
    )

    encoder = LegacyDQNObservationEncoder()

    env.reset()

    state = encoder.encode(
        env
    )

    print(
        f"\nInitial observation: "
        f"{state.shape}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    agent = DQNAgent(
        state_size=84,
        action_size=7,
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

    if "state_dict" in checkpoint:
        state_dict = checkpoint[
            "state_dict"
        ]

    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:
        state_dict = checkpoint

    agent.model.load_state_dict(
        state_dict
    )

    agent.model.eval()
    agent.epsilon = 0.0

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("INITIAL ENVIRONMENT STATE")
    print("=" * 70)

    print_object(
        "Scout",
        env.scout
    )

    print_object(
        "Hunter",
        env.hunter
    )

    print_radar_info(
        env
    )

    # --------------------------------------------------------
    # ACTION NAMES
    # --------------------------------------------------------

    action_names = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT",
        4: "STAY",
        5: "JAM_SUPPRESS",
        6: "JAM_DECEIVE",
    }

    # ========================================================
    # STEP LOOP
    # ========================================================

    for step in range(
        1,
        MAX_STEPS + 1
    ):

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = agent.model(
                state_tensor
            ).squeeze(0)

        action = int(
            torch.argmax(
                q_values
            ).item()
        )

        print("\n")
        print("=" * 70)
        print(
            f"STEP {step}"
        )
        print("=" * 70)

        print(
            f"Q-values:"
        )

        for action_id in range(
            ACTION_SIZE
        ):

            print(
                f"  {action_id} "
                f"{action_names[action_id]:12s} : "
                f"{q_values[action_id].item():10.4f}"
            )

        print(
            f"\nSELECTED ACTION: "
            f"{action} "
            f"({action_names[action]})"
        )

        # ----------------------------------------------------
        # BEFORE
        # ----------------------------------------------------

        print("\nBEFORE STEP")

        print_object(
            "Scout",
            env.scout
        )

        print_object(
            "Hunter",
            env.hunter
        )

        print_radar_info(
            env
        )

        # ----------------------------------------------------
        # ENVIRONMENT STEP
        # ----------------------------------------------------

        next_obs, reward, done, info = (
            env.step(action)
        )

        # ----------------------------------------------------
        # AFTER
        # ----------------------------------------------------

        print("\nAFTER STEP")

        print_object(
            "Scout",
            env.scout
        )

        print_object(
            "Hunter",
            env.hunter
        )

        print_radar_info(
            env
        )

        # ----------------------------------------------------
        # INFO
        # ----------------------------------------------------

        print("\nSTEP INFO")

        print(
            f"Reward           : {reward}"
        )

        print(
            f"Scout Reward     : "
            f"{info.get('reward_scout')}"
        )

        print(
            f"Hunter Reward    : "
            f"{info.get('reward_hunter')}"
        )

        print(
            f"Done             : {done}"
        )

        print(
            f"Radar Failure    : "
            f"{info.get('radar_failure')}"
        )

        print(
            f"Mission Success  : "
            f"{info.get('mission_success')}"
        )

        print(
            f"Mission Failed   : "
            f"{info.get('mission_failed')}"
        )

        print(
            f"Step in info     : "
            f"{info.get('step')}"
        )

        # ----------------------------------------------------
        # EXTRA INFO
        # ----------------------------------------------------

        print("\nFULL INFO KEYS")

        print(
            list(info.keys())
        )

        # ----------------------------------------------------
        # TERMINATION
        # ----------------------------------------------------

        if done:

            print("\n")
            print("=" * 70)
            print("!!! EPISODE TERMINATED !!!")
            print("=" * 70)

            print(
                f"Termination step : {step}"
            )

            print(
                f"Reward           : {reward}"
            )

            print(
                f"Radar failure    : "
                f"{info.get('radar_failure')}"
            )

            print(
                f"Mission success  : "
                f"{info.get('mission_success')}"
            )

            print(
                f"Mission failed   : "
                f"{info.get('mission_failed')}"
            )

            print("\nFINAL STATE")

            print_object(
                "Scout",
                env.scout
            )

            print_object(
                "Hunter",
                env.hunter
            )

            print_radar_info(
                env
            )

            print("\nDiagnostic finished.")

            break

        # ----------------------------------------------------
        # NEXT STATE
        # ----------------------------------------------------

        state = encoder.encode(
            env
        )

    else:

        print("\n")
        print("=" * 70)
        print(
            "Episode did NOT terminate "
            "within 100 steps."
        )
        print("=" * 70)


if __name__ == "__main__":
    main()
