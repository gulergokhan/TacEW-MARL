import argparse
import random
from collections import Counter

from environment.tactical_env import TacticalEnv
from evaluate_happo import (
    get_ctde_observations,
    load_best_model,
    select_deterministic_actions,
)
from evaluate_tactical_holdout import (
    CLEAR_WEATHER,
    EPISODES_PER_SCENARIO,
    PASS_THRESHOLD,
    SCENARIOS,
)
from marl.algorithms.happo import HAPPO


def evaluate_scenario(
    spec,
    scenario_index,
    use_tactical_guards=True,
):
    env = TacticalEnv.from_scenario(
        spec["scenario"],
        weather=CLEAR_WEATHER,
    )

    happo = HAPPO()
    load_best_model(happo)

    rewards = []
    steps = []
    detections = []
    successes = 0
    failures = Counter()

    for episode in range(EPISODES_PER_SCENARIO):
        random.seed(
            60_000
            + scenario_index * 1_000
            + episode
        )

        env.reset()
        done = False
        total_reward = 0.0
        total_steps = 0
        total_detections = 0

        while not done:
            scout_obs, hunter_obs = (
                get_ctde_observations(env)
            )

            scout_action, hunter_action = (
                select_deterministic_actions(
                    happo,
                    scout_obs,
                    hunter_obs,
                    env=(
                        env
                        if use_tactical_guards
                        else None
                    ),
                )
            )

            _, reward, done, info = env.step(
                scout_action=scout_action,
                hunter_action=hunter_action,
            )

            total_reward += float(reward)
            total_steps += 1

            total_detections += len(
                info.get("radar_detections", [])
            )
            total_detections += len(
                info.get(
                    "hunter_radar_detections",
                    [],
                )
            )

        success = bool(
            info.get("mission_success", False)
        )

        successes += int(success)

        if not success:
            failures[
                info.get(
                    "termination_reason",
                    "unknown",
                )
            ] += 1

        rewards.append(total_reward)
        steps.append(total_steps)
        detections.append(total_detections)

    return {
        "name": spec["name"],
        "success_rate": (
            successes / EPISODES_PER_SCENARIO
        ),
        "average_reward": (
            sum(rewards) / len(rewards)
        ),
        "average_steps": (
            sum(steps) / len(steps)
        ),
        "average_detections": (
            sum(detections) / len(detections)
        ),
        "failures": dict(failures),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate HAPPO on unseen tactical scenarios.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help=(
            "Evaluate actor outputs without tactical execution guards."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("HAPPO Holdout Evaluation")
    print("========================")
    print(
        "Execution       : "
        + (
            "raw HAPPO actors"
            if args.raw
            else "HAPPO actors + tactical guards"
        )
    )

    results = []

    for index, spec in enumerate(SCENARIOS):
        result = evaluate_scenario(
            spec,
            index,
            use_tactical_guards=not args.raw,
        )
        results.append(result)

        passed = (
            result["success_rate"]
            >= PASS_THRESHOLD
        )

        print()
        print(f"Scenario       : {result['name']}")
        print(
            f"Result         : "
            f"{'PASS' if passed else 'FAIL'}"
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
        print(f"Failures       : {result['failures']}")

    passed_count = sum(
        result["success_rate"] >= PASS_THRESHOLD
        for result in results
    )

    print()
    print("Holdout Summary")
    print("---------------")
    print(
        f"Passed scenarios: "
        f"{passed_count}/{len(results)}"
    )


if __name__ == "__main__":
    main()
