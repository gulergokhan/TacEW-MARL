import csv
from pathlib import Path

import torch

from environment.gridworld import GridWorld
from train_dqn import train


SEEDS = [7, 21, 42, 84, 123]
EVALUATION_EPISODES = 100

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIRECTORY = (
    PROJECT_ROOT / "experiment_results_double_dqn_checkpoint"
)

MODELS_DIRECTORY = (
    RESULTS_DIRECTORY / "models"
)

SUMMARY_PATH = (
    RESULTS_DIRECTORY / "seed_summary.csv"
)

HISTORY_PATH = (
    RESULTS_DIRECTORY / "training_history.csv"
)


def evaluate_agent(agent):

    environment = GridWorld()

    agent.model.eval()

    total_rewards = []
    total_steps = 0
    successful_episodes = 0
    detected_episodes = 0

    for _ in range(EVALUATION_EPISODES):

        state = environment.reset()
        done = False

        episode_reward = 0
        episode_steps = 0
        episode_detected = False

        while not done:

            action = agent.select_action(
                state,
                training=False
            )

            next_state, reward, done, info = (
                environment.step(action)
            )

            state = next_state
            episode_reward += reward
            episode_steps += 1

            if info["detected"]:
                episode_detected = True

        total_rewards.append(episode_reward)
        total_steps += episode_steps

        if info["reached_goal"]:
            successful_episodes += 1

        if episode_detected:
            detected_episodes += 1

    average_reward = (
        sum(total_rewards) / EVALUATION_EPISODES
    )

    average_steps = (
        total_steps / EVALUATION_EPISODES
    )

    success_rate = (
        successful_episodes
        / EVALUATION_EPISODES
    ) * 100

    detection_rate = (
        detected_episodes
        / EVALUATION_EPISODES
    ) * 100

    return {
        "average_reward": average_reward,
        "average_steps": average_steps,
        "success_rate": success_rate,
        "detection_rate": detection_rate
    }


def save_summary(results):

    with SUMMARY_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "seed",
            "average_reward",
            "average_steps",
            "success_rate",
            "detection_rate",
            "final_training_reward",
            "final_loss"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)


def save_training_history(history_rows):

    with HISTORY_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "seed",
            "episode",
            "average_reward",
            "success_rate",
            "epsilon",
            "loss"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(history_rows)


def run_experiments():

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    MODELS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    experiment_results = []
    all_history_rows = []

    for seed in SEEDS:

        print(
            f"\nSeed {seed} eğitimi başladı...",
            flush=True
        )

        agent, history = train(
            seed=seed,
            verbose=False
        )

        evaluation = evaluate_agent(agent)

        model_path = (
            MODELS_DIRECTORY
            / f"dqn_seed_{seed}.pth"
        )

        torch.save(
            agent.model.state_dict(),
            model_path
        )

        for index, episode in enumerate(
            history["episode"]
        ):

            all_history_rows.append({
                "seed": seed,
                "episode": episode,
                "average_reward": (
                    history["average_reward"][index]
                ),
                "success_rate": (
                    history["success_rate"][index]
                ),
                "epsilon": (
                    history["epsilon"][index]
                ),
                "loss": (
                    history["loss"][index]
                )
            })

        result = {
            "seed": seed,
            "average_reward": (
                evaluation["average_reward"]
            ),
            "average_steps": (
                evaluation["average_steps"]
            ),
            "success_rate": (
                evaluation["success_rate"]
            ),
            "detection_rate": (
                evaluation["detection_rate"]
            ),
            "final_training_reward": (
                history["average_reward"][-1]
            ),
            "final_loss": (
                history["loss"][-1]
            )
        }

        experiment_results.append(result)

        print(
            f"Seed={seed:3} | "
            f"Reward={evaluation['average_reward']:7.2f} | "
            f"Steps={evaluation['average_steps']:6.2f} | "
            f"Success={evaluation['success_rate']:6.2f}% | "
            f"Detection={evaluation['detection_rate']:6.2f}%",
            flush=True
        )

    save_summary(experiment_results)
    save_training_history(all_history_rows)

    print("\nDeneyler tamamlandı.")
    print(f"Özet: {SUMMARY_PATH}")
    print(f"Eğitim geçmişi: {HISTORY_PATH}")


if __name__ == "__main__":
    run_experiments()
