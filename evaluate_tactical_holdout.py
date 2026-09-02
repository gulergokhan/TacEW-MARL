import random
from collections import Counter

from environment.tactical_env import TacticalEnv
from evaluate_tactical_dqn import DEFAULT_MODEL_PATH, load_agent
from weather.models import WeatherState


EPISODES_PER_SCENARIO = 30
PASS_THRESHOLD = 0.80

CLEAR_WEATHER = WeatherState(
    temperature=20.0,
    wind_speed=5.0,
    wind_direction=90.0,
    visibility=20000.0,
    precipitation=0.0,
    cloud_cover=10.0,
)

SHIFTED_RADARS_A = [
    {"radar_id": "radar_01", "x": 4, "y": 4, "detection_range": 3.0},
    {"radar_id": "radar_02", "x": 2, "y": 5, "detection_range": 2.5},
    {"radar_id": "radar_03", "x": 8, "y": 3, "detection_range": 2.5},
]

SHIFTED_RADARS_B = [
    {"radar_id": "radar_01", "x": 5, "y": 2, "detection_range": 3.0},
    {"radar_id": "radar_02", "x": 3, "y": 8, "detection_range": 2.5},
    {"radar_id": "radar_03", "x": 7, "y": 5, "detection_range": 2.5},
]

SCENARIOS = [
    {
        "name": "unseen_start_east",
        "kind": "holdout",
        "scenario": {
            "scout_start": [9, 7],
            "hunter_start": [9, 8],
        },
    },
    {
        "name": "unseen_start_center",
        "kind": "holdout",
        "scenario": {
            "scout_start": [5, 8],
            "hunter_start": [6, 8],
        },
    },
    {
        "name": "shifted_radars",
        "kind": "holdout",
        "scenario": {
            "scout_start": [7, 7],
            "hunter_start": [8, 8],
            "radars": SHIFTED_RADARS_A,
        },
    },
    {
        "name": "unseen_start_and_radars",
        "kind": "holdout",
        "scenario": {
            "scout_start": [9, 6],
            "hunter_start": [9, 7],
            "radars": SHIFTED_RADARS_B,
        },
    },
    {
        "name": "unseen_shifted_target",
        "kind": "holdout",
        "scenario": {
            "scout_start": [7, 7],
            "hunter_start": [8, 8],
            "strike_point": [8, 1],
        },
    },
]


def evaluate_scenario(spec, scenario_index):
    env = TacticalEnv.from_scenario(
        spec["scenario"],
        weather=CLEAR_WEATHER,
    )
    agent = load_agent(env, DEFAULT_MODEL_PATH)

    rewards = []
    steps = []
    detections = []
    successes = 0
    failures = Counter()

    for episode in range(EPISODES_PER_SCENARIO):
        random.seed(50_000 + scenario_index * 1_000 + episode)

        observation = env.reset()
        done = False
        total_reward = 0.0
        total_steps = 0
        total_detections = 0

        while not done:
            action = agent.select_action(observation, training=False)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            total_steps += 1
            total_detections += len(info.get("radar_detections", []))
            total_detections += len(info.get("hunter_radar_detections", []))

        success = bool(info.get("mission_success", False))
        successes += int(success)
        if not success:
            failures[info.get("termination_reason", "unknown")] += 1

        rewards.append(total_reward)
        steps.append(total_steps)
        detections.append(total_detections)

    return {
        "name": spec["name"],
        "kind": spec["kind"],
        "success_rate": successes / EPISODES_PER_SCENARIO,
        "average_reward": sum(rewards) / len(rewards),
        "average_steps": sum(steps) / len(steps),
        "average_detections": sum(detections) / len(detections),
        "failures": dict(failures),
    }


def main():
    print("Tactical DQN Holdout Evaluation")
    print("================================")

    holdout_results = []

    for index, spec in enumerate(SCENARIOS):
        result = evaluate_scenario(spec, index)
        if result["kind"] == "holdout":
            holdout_results.append(result)

        label = "PASS" if result["success_rate"] >= PASS_THRESHOLD else "FAIL"
        if result["kind"] == "diagnostic":
            label = "DIAGNOSTIC"

        print()
        print(f"Scenario       : {result['name']}")
        print(f"Type           : {result['kind']}")
        print(f"Result         : {label}")
        print(f"Success Rate   : {result['success_rate']:.2%}")
        print(f"Average Reward : {result['average_reward']:.2f}")
        print(f"Average Steps  : {result['average_steps']:.2f}")
        print(f"Avg Detections : {result['average_detections']:.2f}")
        print(f"Failures       : {result['failures']}")

    passed = sum(
        result["success_rate"] >= PASS_THRESHOLD
        for result in holdout_results
    )
    print()
    print("Holdout Summary")
    print("---------------")
    print(f"Passed scenarios: {passed}/{len(holdout_results)}")
    print(
        "Decision        : "
        + ("READY FOR MARL" if passed == len(holdout_results) else "GENERALIZATION FIX NEEDED")
    )

if __name__ == "__main__":
    main()
