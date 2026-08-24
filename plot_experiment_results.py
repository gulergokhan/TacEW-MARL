import csv
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIRECTORY = (
    PROJECT_ROOT / "experiment_results"
)

SUMMARY_PATH = (
    RESULTS_DIRECTORY / "seed_summary.csv"
)

HISTORY_PATH = (
    RESULTS_DIRECTORY / "training_history.csv"
)

PLOTS_DIRECTORY = (
    RESULTS_DIRECTORY / "plots"
)

MATPLOTLIB_CACHE = (
    RESULTS_DIRECTORY / ".matplotlib_cache"
)

PLOTS_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

MATPLOTLIB_CACHE.mkdir(
    parents=True,
    exist_ok=True
)

os.environ["MPLCONFIGDIR"] = str(
    MATPLOTLIB_CACHE
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def read_training_history():

    histories = {}

    with HISTORY_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            seed = int(row["seed"])

            if seed not in histories:
                histories[seed] = {
                    "episode": [],
                    "average_reward": [],
                    "success_rate": [],
                    "epsilon": [],
                    "loss": []
                }

            histories[seed]["episode"].append(
                int(row["episode"])
            )

            histories[seed]["average_reward"].append(
                float(row["average_reward"])
            )

            histories[seed]["success_rate"].append(
                float(row["success_rate"])
            )

            histories[seed]["epsilon"].append(
                float(row["epsilon"])
            )

            histories[seed]["loss"].append(
                float(row["loss"])
            )

    return histories


def read_seed_summary():

    results = []

    with SUMMARY_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            results.append({
                "seed": int(row["seed"]),
                "average_reward": float(
                    row["average_reward"]
                ),
                "average_steps": float(
                    row["average_steps"]
                ),
                "success_rate": float(
                    row["success_rate"]
                ),
                "detection_rate": float(
                    row["detection_rate"]
                )
            })

    return results


def plot_training_curves(histories):

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10)
    )

    for seed, history in histories.items():

        label = f"Seed {seed}"

        axes[0, 0].plot(
            history["episode"],
            history["average_reward"],
            label=label
        )

        axes[0, 1].plot(
            history["episode"],
            history["success_rate"],
            label=label
        )

        axes[1, 0].plot(
            history["episode"],
            history["loss"],
            label=label
        )

        axes[1, 1].plot(
            history["episode"],
            history["epsilon"],
            label=label
        )

    axes[0, 0].set_title(
        "Average Reward"
    )
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].axhline(
        0,
        color="black",
        linewidth=0.8
    )

    axes[0, 1].set_title(
        "Training Success Rate"
    )
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Success Rate (%)")
    axes[0, 1].set_ylim(-5, 105)

    axes[1, 0].set_title(
        "Training Loss"
    )
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Loss")

    axes[1, 1].set_title(
        "Epsilon Decay"
    )
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Epsilon")

    for axis in axes.flat:
        axis.grid(
            True,
            alpha=0.3
        )
        axis.legend()

    figure.suptitle(
        "DQN Training Results by Seed",
        fontsize=16
    )

    figure.tight_layout()

    output_path = (
        PLOTS_DIRECTORY
        / "training_curves.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(figure)

    return output_path


def plot_seed_comparison(results):

    seeds = [
        result["seed"]
        for result in results
    ]

    rewards = [
        result["average_reward"]
        for result in results
    ]

    success_rates = [
        result["success_rate"]
        for result in results
    ]

    detection_rates = [
        result["detection_rate"]
        for result in results
    ]

    colors = [
        "seagreen" if reward > 0 else "firebrick"
        for reward in rewards
    ]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    axes[0].bar(
        [str(seed) for seed in seeds],
        rewards,
        color=colors
    )

    axes[0].axhline(
        0,
        color="black",
        linewidth=0.8
    )

    axes[0].set_title(
        "Evaluation Reward by Seed"
    )
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Average Reward")

    positions = list(range(len(seeds)))
    width = 0.35

    axes[1].bar(
        [
            position - width / 2
            for position in positions
        ],
        success_rates,
        width=width,
        label="Success Rate",
        color="seagreen"
    )

    axes[1].bar(
        [
            position + width / 2
            for position in positions
        ],
        detection_rates,
        width=width,
        label="Detection Rate",
        color="firebrick"
    )

    axes[1].set_xticks(
        positions,
        [str(seed) for seed in seeds]
    )

    axes[1].set_ylim(0, 110)
    axes[1].set_title(
        "Success and Detection by Seed"
    )
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Rate (%)")
    axes[1].legend()
    axes[1].grid(
        True,
        axis="y",
        alpha=0.3
    )

    figure.suptitle(
        "DQN Seed Comparison",
        fontsize=16
    )

    figure.tight_layout()

    output_path = (
        PLOTS_DIRECTORY
        / "seed_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(figure)

    return output_path


def main():

    histories = read_training_history()
    results = read_seed_summary()

    training_plot = plot_training_curves(
        histories
    )

    comparison_plot = plot_seed_comparison(
        results
    )

    print("Grafikler oluşturuldu:")
    print(training_plot)
    print(comparison_plot)


if __name__ == "__main__":
    main()
