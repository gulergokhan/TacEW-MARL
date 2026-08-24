import csv
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "experiment_comparison"
)

MATPLOTLIB_CACHE = (
    OUTPUT_DIRECTORY / ".matplotlib_cache"
)

EXPERIMENTS = [
    (
        "Standart DQN",
        PROJECT_ROOT / "experiment_results"
    ),
    (
        "Double DQN",
        PROJECT_ROOT / "experiment_results_double_dqn"
    ),
    (
        "Double DQN\nLR 0.0005",
        PROJECT_ROOT
        / "experiment_results_double_dqn_lr_0005"
    ),
    (
        "Double DQN\nTarget 500",
        PROJECT_ROOT
        / "experiment_results_double_dqn_target_500"
    ),
    (
        "Final\nCheckpoint",
        PROJECT_ROOT
        / "experiment_results_double_dqn_checkpoint"
    )
]

OUTPUT_DIRECTORY.mkdir(
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


def read_summary(directory):

    summary_path = (
        directory / "seed_summary.csv"
    )

    results = []

    with summary_path.open(
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
                "success_rate": float(
                    row["success_rate"]
                ),
                "detection_rate": float(
                    row["detection_rate"]
                )
            })

    return results


def load_experiments():

    experiments = []

    for name, directory in EXPERIMENTS:

        results = read_summary(directory)

        experiments.append({
            "name": name,
            "results": results,
            "mean_reward": (
                sum(
                    result["average_reward"]
                    for result in results
                ) / len(results)
            ),
            "mean_success_rate": (
                sum(
                    result["success_rate"]
                    for result in results
                ) / len(results)
            ),
            "mean_detection_rate": (
                sum(
                    result["detection_rate"]
                    for result in results
                ) / len(results)
            )
        })

    return experiments


def add_bar_labels(axis, bars):

    for bar in bars:

        value = bar.get_height()

        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + (3 if value >= 0 else -10),
            f"{value:.1f}",
            ha="center",
            va=(
                "bottom"
                if value >= 0
                else "top"
            ),
            fontsize=9
        )


def plot_metric_comparison(experiments):

    names = [
        experiment["name"]
        for experiment in experiments
    ]

    rewards = [
        experiment["mean_reward"]
        for experiment in experiments
    ]

    success_rates = [
        experiment["mean_success_rate"]
        for experiment in experiments
    ]

    detection_rates = [
        experiment["mean_detection_rate"]
        for experiment in experiments
    ]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    reward_colors = [
        "seagreen" if reward >= 100
        else "darkorange"
        if reward >= 0
        else "firebrick"
        for reward in rewards
    ]

    reward_bars = axes[0].bar(
        names,
        rewards,
        color=reward_colors
    )

    axes[0].axhline(
        0,
        color="black",
        linewidth=0.8
    )

    axes[0].set_title(
        "Seed'ler Arası Ortalama Reward"
    )
    axes[0].set_ylabel("Average Reward")
    add_bar_labels(axes[0], reward_bars)

    success_bars = axes[1].bar(
        names,
        success_rates,
        color="seagreen"
    )

    axes[1].set_title(
        "Seed Başarı Oranı"
    )
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_ylim(0, 115)
    add_bar_labels(axes[1], success_bars)

    detection_bars = axes[2].bar(
        names,
        detection_rates,
        color="firebrick"
    )

    axes[2].set_title(
        "Radar Tespit Oranı"
    )
    axes[2].set_ylabel("Detection Rate (%)")
    axes[2].set_ylim(0, 115)
    add_bar_labels(
        axes[2],
        detection_bars
    )

    for axis in axes:
        axis.grid(
            True,
            axis="y",
            alpha=0.3
        )
        axis.tick_params(
            axis="x",
            labelsize=9
        )

    figure.suptitle(
        "DQN İyileştirme Deneylerinin Karşılaştırması",
        fontsize=16
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "experiment_metrics.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(figure)

    return output_path


def plot_seed_matrix(experiments):

    seeds = sorted({
        result["seed"]
        for experiment in experiments
        for result in experiment["results"]
    })

    matrix = []

    for experiment in experiments:

        success_by_seed = {
            result["seed"]: result["success_rate"]
            for result in experiment["results"]
        }

        matrix.append([
            success_by_seed[seed]
            for seed in seeds
        ])

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    image = axis.imshow(
        matrix,
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        aspect="auto"
    )

    axis.set_xticks(
        range(len(seeds)),
        [str(seed) for seed in seeds]
    )

    axis.set_yticks(
        range(len(experiments)),
        [
            experiment["name"].replace(
                "\n",
                " "
            )
            for experiment in experiments
        ]
    )

    axis.set_xlabel("Seed")
    axis.set_ylabel("Deney")
    axis.set_title(
        "Seed Bazında Başarı Oranları"
    )

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):

            axis.text(
                column_index,
                row_index,
                f"{value:.0f}%",
                ha="center",
                va="center",
                color=(
                    "white"
                    if value < 50
                    else "black"
                ),
                fontweight="bold"
            )

    colorbar = figure.colorbar(
        image,
        ax=axis
    )

    colorbar.set_label(
        "Success Rate (%)"
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "seed_success_matrix.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(figure)

    return output_path


def main():

    experiments = load_experiments()

    metric_plot = plot_metric_comparison(
        experiments
    )

    matrix_plot = plot_seed_matrix(
        experiments
    )

    print("Karşılaştırma grafikleri oluşturuldu:")
    print(metric_plot)
    print(matrix_plot)


if __name__ == "__main__":
    main()
