import random
from collections import Counter

from environment.tactical_env import TacticalEnv
from evaluate_tactical_dqn import (
    DEFAULT_MODEL_PATH,
    load_agent,
)
from weather.models import WeatherState


EPISODES_PER_SCENARIO = 50


CLEAR_WEATHER = WeatherState(
    temperature=20.0,
    wind_speed=5.0,
    wind_direction=90.0,
    visibility=20000.0,
    precipitation=0.0,
    cloud_cover=10.0,
)

HEAVY_WEATHER = WeatherState(
    temperature=10.0,
    wind_speed=25.0,
    wind_direction=240.0,
    visibility=4000.0,
    precipitation=30.0,
    cloud_cover=90.0,
)


SCENARIOS = [
    {
        "name": "baseline_clear",
        "scout_start": (7, 7),
        "hunter_start": (8, 8),
        "weather": CLEAR_WEATHER,
    },
    {
        "name": "baseline_heavy_weather",
        "scout_start": (7, 7),
        "hunter_start": (8, 8),
        "weather": HEAVY_WEATHER,
    },
    {
        "name": "close_formation",
        "scout_start": (7, 8),
        "hunter_start": (8, 8),
        "weather": CLEAR_WEATHER,
    },
    {
        "name": "wide_formation",
        "scout_start": (6, 7),
        "hunter_start": (8, 8),
        "weather": CLEAR_WEATHER,
    },
    {
        "name": "north_offset",
        "scout_start": (8, 5),
        "hunter_start": (8, 6),
        "weather": CLEAR_WEATHER,
    },
]


def evaluate_scenario(
    scenario,
    scenario_index,
):
    env = TacticalEnv(
        weather=scenario["weather"],
        scout_start_position=(
            scenario["scout_start"]
        ),
        hunter_start_position=(
            scenario["hunter_start"]
        ),
    )

    agent = load_agent(
        env,
        DEFAULT_MODEL_PATH,
    )

    rewards = []
    steps = []
    detections = []
    successes = 0
    failure_reasons = Counter()

    for episode in range(
        EPISODES_PER_SCENARIO
    ):
        random.seed(
            1000
            + scenario_index * 100
            + episode
        )

        observation = env.reset()
        done = False

        total_reward = 0.0
        total_steps = 0
        total_detections = 0

        while not done:
            action = agent.select_action(
                observation,
                training=False,
            )

            (
                observation,
                reward,
                done,
                info,
            ) = env.step(action)

            total_reward += reward
            total_steps += 1

            total_detections += len(
                info.get(
                    "radar_detections",
                    [],
                )
                +
                info.get(
                    "hunter_radar_detections",
                    [],
                )
            )

        success = bool(
            info.get(
                "mission_success",
                False,
            )
        )

        successes += int(success)

        if not success:
            failure_reasons[
                info.get(
                    "termination_reason",
                    "unknown",
                )
            ] += 1

        rewards.append(total_reward)
        steps.append(total_steps)
        detections.append(
            total_detections
        )

    return {
        "name": scenario["name"],
        "success_rate": (
            successes
            / EPISODES_PER_SCENARIO
        ),
        "average_reward": (
            sum(rewards)
            / len(rewards)
        ),
        "average_steps": (
            sum(steps)
            / len(steps)
        ),
        "average_detections": (
            sum(detections)
            / len(detections)
        ),
        "failure_reasons": dict(
            failure_reasons
        ),
    }


def main():
    print(
        "Escort DQN Robustness Evaluation"
    )

    print(
        "================================"
    )

    for index, scenario in enumerate(
        SCENARIOS
    ):
        result = evaluate_scenario(
            scenario,
            index,
        )

        print()
        print(
            f"Scenario       : "
            f"{result['name']}"
        )
        print(
            f"Success Rate   : "
            f"{result['success_rate']:.2%}"
        )
        print(
            f"Average Reward : "
            f"{result['average_reward']:.2f}"
        )
        print(
            f"Average Steps  : "
            f"{result['average_steps']:.2f}"
        )
        print(
            f"Avg Detections : "
            f"{result['average_detections']:.2f}"
        )
        print(
            f"Failures       : "
            f"{result['failure_reasons']}"
        )


if __name__ == "__main__":
    main()
