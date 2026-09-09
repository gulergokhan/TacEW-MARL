import argparse
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from environment.tactical_env import TacticalEnv
from marl.algorithms.happo import HAPPO
from marl.buffer import MARLRolloutBuffer
from marl.execution import (
    apply_hunter_progress_guard,
    apply_scout_jamming_guard,
    get_hunter_expert_action,
    get_tactical_expert_actions,
)
from configs import marl_config as cfg
from dashboard_episode_store import replace_episode_group
from weather.models import WeatherState


CLEAR_TRAINING_WEATHER = WeatherState(
    temperature=20.0,
    wind_speed=0.0,
    wind_direction=0.0,
    visibility=10000.0,
    precipitation=0.0,
    cloud_cover=0.0,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_ctde_observations(env):
    """
    Read decentralized observations and centralized state
    directly from TacticalEnv.
    """

    state = env._get_state()

    observations = env._get_agent_observations(state)

    return (
        observations["scout"],
        observations["hunter"],
        observations["global"],
    )


def get_training_scenario(episode):
    """Return a gradual and target-balanced curriculum scenario."""

    scenario_index = episode - 1
    target_count = len(cfg.TRAINING_STRIKE_POINTS)
    formation_count = len(cfg.TRAINING_START_FORMATIONS)

    target_index = scenario_index % target_count
    strike_point = cfg.TRAINING_STRIKE_POINTS[
        target_index
    ]
    formation_index = (
        scenario_index // target_count
    ) % formation_count
    scout_start, hunter_start = (
        cfg.TRAINING_START_FORMATIONS[formation_index]
    )
    radar_cycle_index = (
        scenario_index
        // (target_count * formation_count)
    ) % len(cfg.HAPPO_TRAINING_RADAR_LAYOUT_INDICES)
    radar_layout_index = (
        cfg.HAPPO_TRAINING_RADAR_LAYOUT_INDICES[
            radar_cycle_index
        ]
    )

    if episode <= cfg.CURRICULUM_STAGE_1_END:
        return (
            1,
            scout_start,
            hunter_start,
            strike_point,
            radar_layout_index,
        )

    if episode <= cfg.CURRICULUM_STAGE_2_END:
        return (
            2,
            scout_start,
            hunter_start,
            strike_point,
            radar_layout_index,
        )

    return (
        3,
        scout_start,
        hunter_start,
        strike_point,
        radar_layout_index,
    )


def get_curriculum_radars(
    curriculum_stage,
    radar_layout_index,
    forbidden_positions=(),
):
    """Increase radar complexity only after route coordination is learned."""

    if curriculum_stage == 1:
        return []

    if curriculum_stage == 2:
        return None

    return build_expert_radar_layout(
        radar_layout_index,
        forbidden_positions=forbidden_positions,
    )


def get_entropy_coefficient(episode):
    """Linearly reduce exploration without turning it off abruptly."""

    denominator = max(
        cfg.NUM_EPISODES - 1,
        1,
    )
    progress = min(
        max((episode - 1) / denominator, 0.0),
        1.0,
    )

    return (
        cfg.ENTROPY_COEF
        + progress
        * (
            cfg.ENTROPY_COEF_END
            - cfg.ENTROPY_COEF
        )
    )


def build_expert_radar_layout(
    layout_index,
    forbidden_positions=(),
):
    """Create deterministic synthetic radar layouts without holdout leakage."""

    if layout_index == 0:
        return None

    generator = random.Random(
        cfg.SEED + 300_000 + layout_index
    )
    forbidden = set(forbidden_positions)
    available_positions = [
        (x, y)
        for x in range(cfg.TRAINING_GRID_SIZE)
        for y in range(cfg.TRAINING_GRID_SIZE)
        if (x, y) not in forbidden
    ]
    positions = generator.sample(
        available_positions,
        3,
    )
    detection_ranges = (3.0, 2.5, 2.5)

    return [
        {
            "radar_id": f"radar_{index + 1:02d}",
            "x": position[0],
            "y": position[1],
            "detection_range": detection_ranges[index],
        }
        for index, position in enumerate(positions)
    ]


def collect_expert_demonstrations():
    """Collect successful coordinated trajectories for actor warm-up."""

    clear_weather = WeatherState(
        temperature=20.0,
        wind_speed=0.0,
        wind_direction=0.0,
        visibility=10000.0,
        precipitation=0.0,
        cloud_cover=0.0,
    )

    scout_observations = []
    hunter_observations = []
    scout_actions = []
    hunter_actions = []
    successful_episodes = 0
    attempted_episodes = 0

    for target_index, strike_point in enumerate(
        cfg.TRAINING_STRIKE_POINTS
    ):
        for formation_index, formation in enumerate(
            cfg.TRAINING_START_FORMATIONS
        ):
            scout_start, hunter_start = formation

            layout_indices = list(
                range(cfg.EXPERT_RADAR_LAYOUT_COUNT)
            )

            if cfg.EXPERT_INCLUDE_NO_RADAR_LAYOUT:
                layout_indices.insert(0, -1)

            for layout_index in layout_indices:
                radars_config = (
                    []
                    if layout_index == -1
                    else build_expert_radar_layout(
                        layout_index,
                        forbidden_positions=(
                            scout_start,
                            hunter_start,
                            strike_point,
                        ),
                    )
                )

                repeat_count = (
                    cfg.EXPERT_DEMONSTRATION_REPEATS
                    * (
                        cfg.EXPERT_DEFAULT_LAYOUT_REPEAT_MULTIPLIER
                        if layout_index == 0
                        else 1
                    )
                )

                for repeat in range(
                    repeat_count
                ):
                    attempted_episodes += 1
                    random.seed(
                        cfg.SEED
                        + 200_000
                        + target_index * 100_000
                        + formation_index * 10_000
                        + (layout_index + 1) * 100
                        + repeat
                    )

                    env = TacticalEnv(
                        weather=clear_weather,
                        scout_start_position=scout_start,
                        hunter_start_position=hunter_start,
                        strike_point=strike_point,
                        radars_config=radars_config,
                    )
                    env.reset()
                    episode_samples = []
                    done = False
                    info = {}

                    while not done:
                        scout_obs, hunter_obs, _ = (
                            get_ctde_observations(env)
                        )
                        scout_action, hunter_action = (
                            get_tactical_expert_actions(
                                env
                            )
                        )

                        episode_samples.append(
                            (
                                scout_obs,
                                hunter_obs,
                                scout_action,
                                hunter_action,
                            )
                        )

                        (
                            _,
                            _,
                            done,
                            info,
                        ) = env.step(
                            scout_action=scout_action,
                            hunter_action=hunter_action,
                        )

                    if not info.get("mission_success", False):
                        continue

                    successful_episodes += 1

                    for (
                        scout_obs,
                        hunter_obs,
                        scout_action,
                        hunter_action,
                    ) in episode_samples:
                        scout_observations.append(scout_obs)
                        hunter_observations.append(hunter_obs)
                        scout_actions.append(scout_action)
                        hunter_actions.append(hunter_action)

    if not scout_observations:
        raise RuntimeError(
            "Expert demonstration collection produced no successful episodes."
        )

    return {
        "scout_observations": np.asarray(
            scout_observations,
            dtype=np.float32,
        ),
        "hunter_observations": np.asarray(
            hunter_observations,
            dtype=np.float32,
        ),
        "scout_actions": np.asarray(
            scout_actions,
            dtype=np.int64,
        ),
        "hunter_actions": np.asarray(
            hunter_actions,
            dtype=np.int64,
        ),
        "successful_episodes": successful_episodes,
        "attempted_episodes": attempted_episodes,
        "radar_layouts": len(layout_indices),
    }


def pretrain_actor(
    actor,
    observations,
    actions,
    device,
    epochs=None,
):
    """Fit one decentralized actor to successful expert actions."""

    observation_tensor = torch.as_tensor(
        observations,
        dtype=torch.float32,
        device=device,
    )
    action_tensor = torch.as_tensor(
        actions,
        dtype=torch.long,
        device=device,
    )
    optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=cfg.EXPERT_PRETRAINING_LR,
    )
    actor.train()

    epoch_count = (
        cfg.EXPERT_PRETRAINING_EPOCHS
        if epochs is None
        else int(epochs)
    )

    for _ in range(epoch_count):
        indices = torch.randperm(
            observation_tensor.shape[0],
            device=device,
        )

        for start in range(
            0,
            observation_tensor.shape[0],
            cfg.EXPERT_PRETRAINING_BATCH_SIZE,
        ):
            batch_indices = indices[
                start:
                start + cfg.EXPERT_PRETRAINING_BATCH_SIZE
            ]
            logits = actor(
                observation_tensor[batch_indices]
            )
            loss = F.cross_entropy(
                logits,
                action_tensor[batch_indices],
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

    actor.eval()

    with torch.no_grad():
        predictions = actor(
            observation_tensor
        ).argmax(dim=-1)
        accuracy = (
            predictions == action_tensor
        ).float().mean().item()

    actor.train()

    return accuracy * 100.0


def train_actors_from_demonstrations(
    happo,
    demonstrations,
    epochs,
):
    """Apply the same balanced expert batch to both role policies."""

    scout_accuracy = pretrain_actor(
        happo.scout_actor,
        demonstrations["scout_observations"],
        demonstrations["scout_actions"],
        happo.device,
        epochs=epochs,
    )
    hunter_accuracy = pretrain_actor(
        happo.hunter_actor,
        demonstrations["hunter_observations"],
        demonstrations["hunter_actions"],
        happo.device,
        epochs=epochs,
    )

    return scout_accuracy, hunter_accuracy


def pretrain_actors_from_expert(happo):
    """Warm-start both actors from successful role-specific demonstrations."""

    demonstrations = collect_expert_demonstrations()

    scout_accuracy, hunter_accuracy = (
        train_actors_from_demonstrations(
            happo,
            demonstrations,
            epochs=cfg.EXPERT_PRETRAINING_EPOCHS,
        )
    )

    return {
        "samples": len(demonstrations["scout_actions"]),
        "successful_episodes": demonstrations[
            "successful_episodes"
        ],
        "attempted_episodes": demonstrations[
            "attempted_episodes"
        ],
        "radar_layouts": demonstrations["radar_layouts"],
        "scout_accuracy": scout_accuracy,
        "hunter_accuracy": hunter_accuracy,
    }


def run_guided_scratch_warmup(happo):
    """Teach target-conditioned actions without loading an old checkpoint."""

    demonstrations = collect_expert_demonstrations()
    best_score = (
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -float("inf"),
    )
    best_validation = None
    best_scout_state = None
    best_hunter_state = None
    best_epoch = 0

    completed_epochs = 0

    while completed_epochs < cfg.SCRATCH_GUIDED_MAX_EPOCHS:
        block_epochs = min(
            cfg.SCRATCH_GUIDED_BLOCK_EPOCHS,
            cfg.SCRATCH_GUIDED_MAX_EPOCHS
            - completed_epochs,
        )
        scout_accuracy, hunter_accuracy = (
            train_actors_from_demonstrations(
                happo,
                demonstrations,
                epochs=block_epochs,
            )
        )
        completed_epochs += block_epochs

        validation = evaluate_training_targets(happo)
        score = validation_score(validation)

        if score > best_score:
            best_score = score
            best_validation = validation
            best_scout_state = deepcopy(
                happo.scout_actor.state_dict()
            )
            best_hunter_state = deepcopy(
                happo.hunter_actor.state_dict()
            )
            best_epoch = completed_epochs

        print(
            "[GUIDED SCRATCH] "
            f"epochs={completed_epochs} | "
            f"Success Rate={validation['mean_success_rate']:.2f}% | "
            f"Worst Success Rate={validation['worst_success_rate']:.2f}% | "
            f"Radar Success Rate={validation['radar_success_rate']:.2f}% | "
            f"Worst Radar={validation['worst_radar_success_rate']:.2f}% | "
            f"scout_accuracy={scout_accuracy:.2f}% | "
            f"hunter_accuracy={hunter_accuracy:.2f}%"
        )

        if (
            validation["worst_success_rate"]
            >= cfg.EARLY_STOP_WORST_SUCCESS
            and validation["worst_radar_success_rate"]
            >= cfg.EARLY_STOP_WORST_SUCCESS
        ):
            break

    if best_validation is None:
        raise RuntimeError(
            "Guided scratch warm-up produced no validation result."
        )

    happo.scout_actor.load_state_dict(best_scout_state)
    happo.hunter_actor.load_state_dict(best_hunter_state)

    return {
        "demonstrations": demonstrations,
        "validation": best_validation,
        "score": best_score,
        "epochs": best_epoch,
    }


def select_deterministic_actions(
    happo,
    scout_observation,
    hunter_observation,
    env=None,
):
    scout_tensor = torch.as_tensor(
        scout_observation,
        dtype=torch.float32,
        device=happo.device,
    ).unsqueeze(0)
    hunter_tensor = torch.as_tensor(
        hunter_observation,
        dtype=torch.float32,
        device=happo.device,
    ).unsqueeze(0)

    with torch.no_grad():
        scout_distribution = (
            happo.scout_actor.get_action_distribution(
                scout_tensor
            )
        )
        hunter_distribution = (
            happo.hunter_actor.get_action_distribution(
                hunter_tensor
            )
        )

    scout_action = int(
        torch.argmax(scout_distribution.probs).item()
    )
    hunter_action = int(
        torch.argmax(hunter_distribution.probs).item()
    )

    if env is not None:
        scout_action = apply_scout_jamming_guard(
            env,
            scout_action,
        )
        hunter_action = apply_hunter_progress_guard(
            env,
            scout_action,
            hunter_action,
        )

    return scout_action, hunter_action


def collect_dashboard_episode(happo, label):
    """Create one guarded deterministic sortie for dashboard playback."""

    env = TacticalEnv(weather=CLEAR_TRAINING_WEATHER)
    env.reset()
    done = False

    while not done:
        scout_obs, hunter_obs, _ = get_ctde_observations(env)
        scout_action, hunter_action = select_deterministic_actions(
            happo,
            scout_obs,
            hunter_obs,
            env=env,
        )
        _, _, done, _ = env.step(
            scout_action=scout_action,
            hunter_action=hunter_action,
        )

    return {
        "label": label,
        "steps": deepcopy(env.episode_log),
    }


def publish_happo_dashboard_episodes(
    happo,
    training_episodes,
    training_mode,
    completed_episodes,
):
    """Publish sampled training sorties plus the restored final policy."""

    final_episode = collect_dashboard_episode(
        happo,
        (
            "HAPPO Final Policy + Tactical Guard "
            f"({training_mode}, episode {completed_episodes})"
        ),
    )
    episodes = list(training_episodes) + [final_episode]
    replace_episode_group(
        episodes,
        (
            "HAPPO Training Episode",
            "HAPPO Final Policy",
        ),
    )

    print(
        "Dashboard playback updated with "
        f"{len(episodes)} HAPPO episode(s)."
    )


def validation_score(metrics):
    """Rank checkpoints by robustness before average performance."""
    radar_success_rate = metrics.get(
        "radar_success_rate",
        metrics["worst_success_rate"],
    )
    worst_radar_success_rate = metrics.get(
        "worst_radar_success_rate",
        radar_success_rate,
    )

    return (
        min(
            metrics["worst_success_rate"],
            worst_radar_success_rate,
        ),
        worst_radar_success_rate,
        metrics["worst_success_rate"],
        radar_success_rate,
        metrics["mean_success_rate"],
        metrics["average_reward"],
    )


def evaluate_training_targets(happo):
    """Evaluate every training target and formation deterministically."""

    random_state = random.getstate()
    scout_was_training = happo.scout_actor.training
    hunter_was_training = happo.hunter_actor.training

    happo.scout_actor.eval()
    happo.hunter_actor.eval()

    target_results = {}
    radar_results = {}
    all_rewards = []

    clear_weather = WeatherState(
        temperature=20.0,
        wind_speed=0.0,
        wind_direction=0.0,
        visibility=10000.0,
        precipitation=0.0,
        cloud_cover=0.0,
    )

    try:
        for target_index, strike_point in enumerate(
            cfg.TRAINING_STRIKE_POINTS
        ):
            successes = 0
            attempts = 0
            rewards = []

            for formation_index, formation in enumerate(
                cfg.TRAINING_START_FORMATIONS
            ):
                scout_start, hunter_start = formation

                for repeat in range(
                    cfg.VALIDATION_EPISODES_PER_CASE
                ):
                    random.seed(
                        cfg.SEED
                        + 100_000
                        + target_index * 10_000
                        + formation_index * 100
                        + repeat
                    )

                    env = TacticalEnv(
                        weather=clear_weather,
                        scout_start_position=scout_start,
                        hunter_start_position=hunter_start,
                        strike_point=strike_point,
                    )
                    env.reset()

                    done = False
                    total_reward = 0.0
                    info = {}

                    while not done:
                        scout_obs, hunter_obs, _ = (
                            get_ctde_observations(env)
                        )
                        scout_action, hunter_action = (
                            select_deterministic_actions(
                                happo,
                                scout_obs,
                                hunter_obs,
                            )
                        )
                        _, reward, done, info = env.step(
                            scout_action=scout_action,
                            hunter_action=hunter_action,
                        )
                        total_reward += float(reward)

                    successes += int(
                        info.get("mission_success", False)
                    )
                    attempts += 1
                    rewards.append(total_reward)
                    all_rewards.append(total_reward)

            target_key = (
                f"{strike_point[0]},{strike_point[1]}"
            )
            target_results[target_key] = {
                "success_rate": (
                    successes / attempts * 100.0
                ),
                "average_reward": float(
                    np.mean(rewards)
                ),
            }

        radar_strike_point = cfg.TRAINING_STRIKE_POINTS[0]

        for layout_index in (
            cfg.RADAR_VALIDATION_LAYOUT_INDICES
        ):
            for formation_index, formation in enumerate(
                cfg.TRAINING_START_FORMATIONS
            ):
                radar_scout_start, radar_hunter_start = formation
                successes = 0
                rewards = []
                radars_config = build_expert_radar_layout(
                    layout_index,
                    forbidden_positions=(
                        radar_scout_start,
                        radar_hunter_start,
                        radar_strike_point,
                    ),
                )

                for repeat in range(
                    cfg.RADAR_VALIDATION_EPISODES
                ):
                    random.seed(
                        cfg.SEED
                        + 400_000
                        + layout_index * 1_000
                        + formation_index * 100
                        + repeat
                    )
                    env = TacticalEnv(
                        weather=clear_weather,
                        scout_start_position=radar_scout_start,
                        hunter_start_position=radar_hunter_start,
                        strike_point=radar_strike_point,
                        radars_config=radars_config,
                    )
                    env.reset()
                    done = False
                    total_reward = 0.0
                    info = {}

                    while not done:
                        scout_obs, hunter_obs, _ = (
                            get_ctde_observations(env)
                        )
                        scout_action, hunter_action = (
                            select_deterministic_actions(
                                happo,
                                scout_obs,
                                hunter_obs,
                            )
                        )
                        _, reward, done, info = env.step(
                            scout_action=scout_action,
                            hunter_action=hunter_action,
                        )
                        total_reward += float(reward)

                    successes += int(
                        info.get("mission_success", False)
                    )
                    rewards.append(total_reward)
                    all_rewards.append(total_reward)

                result_key = (
                    f"{layout_index}:{formation_index}"
                )
                radar_results[result_key] = {
                    "success_rate": (
                        successes
                        / cfg.RADAR_VALIDATION_EPISODES
                        * 100.0
                    ),
                    "average_reward": float(
                        np.mean(rewards)
                    ),
                }
    finally:
        random.setstate(random_state)

        if scout_was_training:
            happo.scout_actor.train()
        if hunter_was_training:
            happo.hunter_actor.train()

    success_rates = [
        result["success_rate"]
        for result in target_results.values()
    ]
    radar_success_rates = [
        result["success_rate"]
        for result in radar_results.values()
    ]

    return {
        "targets": target_results,
        "radar_layouts": radar_results,
        "worst_success_rate": min(success_rates),
        "mean_success_rate": float(
            np.mean(success_rates)
        ),
        "radar_success_rate": float(
            np.mean(radar_success_rates)
        ),
        "worst_radar_success_rate": min(
            radar_success_rates
        ),
        "average_reward": float(
            np.mean(all_rewards)
        ),
    }


def evaluate_existing_best(model_path):
    """Score the existing best model so a weaker run cannot replace it."""

    path = Path(model_path)

    if not path.exists():
        return None

    try:
        checkpoint = torch.load(
            path,
            map_location=cfg.DEVICE,
            weights_only=True,
        )
        candidate = HAPPO()
        candidate.scout_actor.load_state_dict(
            checkpoint["scout_actor"]
        )
        candidate.hunter_actor.load_state_dict(
            checkpoint["hunter_actor"]
        )

        if "critic" in checkpoint:
            candidate.critic.load_state_dict(
                checkpoint["critic"]
            )

        return evaluate_training_targets(candidate)
    except (
        OSError,
        KeyError,
        RuntimeError,
    ) as error:
        print(
            "Existing best model could not be validated: "
            f"{error}"
        )
        return None


def get_training_paths(training_mode):
    """Keep scratch experiments separate from the production model."""

    if training_mode == "scratch":
        return {
            "best": Path("models/happo_scratch_best.pth"),
            "final": Path("models/happo_scratch_final.pth"),
            "checkpoint_prefix": "happo_scratch_episode",
        }

    if training_mode == "resume":
        return {
            "best": Path("models/happo_best.pth"),
            "final": Path("models/happo_final.pth"),
            "checkpoint_prefix": "happo_episode",
        }

    raise ValueError(
        "Training mode must be scratch or resume."
    )


def main(training_mode="scratch"):
    training_paths = get_training_paths(training_mode)
    resume_best = training_mode == "resume"

    set_seed(cfg.SEED)
    print("=" * 60)
    print("TALON - HAPPO / CTDE TRAINING")
    print("=" * 60)

    print(f"Scout observation : {cfg.SCOUT_OBS_DIM}")
    print(f"Hunter observation: {cfg.HUNTER_OBS_DIM}")
    print(f"Global state     : {cfg.GLOBAL_STATE_DIM}")
    print(f"Scout actions     : {cfg.SCOUT_ACTION_DIM}")
    print(f"Hunter actions    : {cfg.HUNTER_ACTION_DIM}")
    print(f"Episodes         : {cfg.NUM_EPISODES}")
    print(f"Device           : {cfg.DEVICE}")
    print(f"Training mode    : {training_mode}")
    print(f"Best checkpoint  : {training_paths['best']}")
    print("=" * 60)

    happo = HAPPO()
    buffer = MARLRolloutBuffer()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    best_model_path = training_paths["best"]
    final_model_path = training_paths["final"]
    checkpoint_prefix = training_paths[
        "checkpoint_prefix"
    ]

    episode_rewards = []
    episode_successes = []
    episode_scout_rewards = []
    episode_hunter_rewards = []
    episode_step_counts = []
    best_worst_success_rate = -1.0
    best_success_rate = -1.0
    best_radar_success_rate = -1.0
    best_worst_radar_success_rate = -1.0
    best_reward = -float("inf")
    best_validation_score = (
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -float("inf"),
    )
    perfect_validation_streak = 0
    completed_episodes = 0
    latest_update_metrics = None
    happo_update_count = 0
    scratch_demonstrations = None
    dashboard_training_episodes = []

    existing_best = None

    if resume_best:
        existing_best = evaluate_existing_best(
            best_model_path
        )

    if existing_best is not None:
        existing_checkpoint = torch.load(
            best_model_path,
            map_location=happo.device,
            weights_only=True,
        )
        happo.scout_actor.load_state_dict(
            existing_checkpoint["scout_actor"]
        )
        happo.hunter_actor.load_state_dict(
            existing_checkpoint["hunter_actor"]
        )

        if "critic" in existing_checkpoint:
            happo.critic.load_state_dict(
                existing_checkpoint["critic"]
            )

        best_validation_score = validation_score(
            existing_best
        )
        best_worst_success_rate = float(
            existing_best["worst_success_rate"]
        )
        best_success_rate = float(
            existing_best["mean_success_rate"]
        )
        best_radar_success_rate = float(
            existing_best["radar_success_rate"]
        )
        best_worst_radar_success_rate = float(
            existing_best["worst_radar_success_rate"]
        )
        best_reward = float(
            existing_best["average_reward"]
        )

        print(
            "Existing best baseline | "
            f"Success Rate={best_success_rate:.2f}% | "
            f"Worst Success Rate={best_worst_success_rate:.2f}% | "
            f"Radar Success Rate={best_radar_success_rate:.2f}% | "
            f"Worst Radar={best_worst_radar_success_rate:.2f}% | "
            f"reward={best_reward:.2f}"
        )
        print(
            "Existing HAPPO checkpoint loaded for "
            "expert repair."
        )

    if (
        not resume_best
        and cfg.SCRATCH_GUIDED_WARMUP_ENABLED
    ):
        guided_warmup = run_guided_scratch_warmup(
            happo
        )
        scratch_demonstrations = guided_warmup[
            "demonstrations"
        ]
        warm_start_validation = guided_warmup[
            "validation"
        ]
        best_validation_score = guided_warmup[
            "score"
        ]
        best_worst_success_rate = float(
            warm_start_validation["worst_success_rate"]
        )
        best_success_rate = float(
            warm_start_validation["mean_success_rate"]
        )
        best_radar_success_rate = float(
            warm_start_validation["radar_success_rate"]
        )
        best_worst_radar_success_rate = float(
            warm_start_validation[
                "worst_radar_success_rate"
            ]
        )
        best_reward = float(
            warm_start_validation["average_reward"]
        )

        guided_checkpoint = {
            "episode": 0,
            "training_phase": "guided_scratch_warmup",
            "guided_epochs": guided_warmup["epochs"],
            "scout_actor": happo.scout_actor.state_dict(),
            "hunter_actor": happo.hunter_actor.state_dict(),
            "critic": happo.critic.state_dict(),
            "best_worst_success_rate": (
                best_worst_success_rate
            ),
            "best_success_rate": best_success_rate,
            "best_radar_success_rate": (
                best_radar_success_rate
            ),
            "best_worst_radar_success_rate": (
                best_worst_radar_success_rate
            ),
            "best_average_reward": best_reward,
            "validation_targets": (
                warm_start_validation["targets"]
            ),
            "validation_radar_layouts": (
                warm_start_validation["radar_layouts"]
            ),
        }
        torch.save(
            guided_checkpoint,
            best_model_path,
        )
        print(
            "-> Guided scratch baseline saved "
            f"(epochs={guided_warmup['epochs']} | "
            f"Success Rate={best_success_rate:.2f}% | "
            f"Worst Success Rate={best_worst_success_rate:.2f}% | "
            f"Radar Success Rate={best_radar_success_rate:.2f}% | "
            f"Worst Radar={best_worst_radar_success_rate:.2f}%)"
        )

    if resume_best and cfg.EXPERT_PRETRAINING_ENABLED:
        # Existing-checkpoint validation constructs a temporary model and
        # consumes Torch RNG state. Reset here so the expert warm start is
        # reproducible whether or not an older checkpoint exists.
        set_seed(cfg.SEED)

        pretraining = pretrain_actors_from_expert(happo)

        print(
            "Expert warm start | "
            f"episodes={pretraining['successful_episodes']}/"
            f"{pretraining['attempted_episodes']} | "
            f"radar_layouts={pretraining['radar_layouts']} | "
            f"samples={pretraining['samples']} | "
            f"scout_accuracy={pretraining['scout_accuracy']:.2f}% | "
            f"hunter_accuracy={pretraining['hunter_accuracy']:.2f}%"
        )

        warm_start_validation = evaluate_training_targets(
            happo
        )
        warm_start_score = validation_score(
            warm_start_validation
        )
        warm_target_summary = " | ".join(
            f"{target}={result['success_rate']:.0f}%"
            for target, result
            in warm_start_validation["targets"].items()
        )

        print(
            "[WARM START VALIDATION] "
            f"Success Rate={warm_start_validation['mean_success_rate']:.2f}% | "
            f"Worst Success Rate={warm_start_validation['worst_success_rate']:.2f}% | "
            f"Radar Success Rate={warm_start_validation['radar_success_rate']:.2f}% | "
            f"Worst Radar={warm_start_validation['worst_radar_success_rate']:.2f}% | "
            f"reward={warm_start_validation['average_reward']:.2f} | "
            f"{warm_target_summary}"
        )

        if warm_start_score > best_validation_score:
            best_validation_score = warm_start_score
            best_worst_success_rate = float(
                warm_start_validation["worst_success_rate"]
            )
            best_success_rate = float(
                warm_start_validation["mean_success_rate"]
            )
            best_radar_success_rate = float(
                warm_start_validation["radar_success_rate"]
            )
            best_worst_radar_success_rate = float(
                warm_start_validation[
                    "worst_radar_success_rate"
                ]
            )
            best_reward = float(
                warm_start_validation["average_reward"]
            )

            warm_start_checkpoint = {
                "episode": 0,
                "training_phase": "expert_repair",
                "scout_actor": (
                    happo.scout_actor.state_dict()
                ),
                "hunter_actor": (
                    happo.hunter_actor.state_dict()
                ),
                "critic": happo.critic.state_dict(),
                "best_worst_success_rate": (
                    best_worst_success_rate
                ),
                "best_success_rate": best_success_rate,
                "best_radar_success_rate": (
                    best_radar_success_rate
                ),
                "best_worst_radar_success_rate": (
                    best_worst_radar_success_rate
                ),
                "best_average_reward": best_reward,
                "validation_targets": (
                    warm_start_validation["targets"]
                ),
                "validation_radar_layouts": (
                    warm_start_validation["radar_layouts"]
                ),
            }

            torch.save(
                warm_start_checkpoint,
                best_model_path,
            )

            print(
                "-> Expert warm-start checkpoint saved "
                f"(Success Rate={best_success_rate:.2f}% | "
                f"Worst Success Rate={best_worst_success_rate:.2f}% | "
                f"Radar Success Rate={best_radar_success_rate:.2f}% | "
                f"Worst Radar={best_worst_radar_success_rate:.2f}% | "
                f"avg={best_reward:.2f})"
            )

            if (
                best_worst_success_rate
                >= cfg.EARLY_STOP_WORST_SUCCESS
                and best_worst_radar_success_rate
                >= cfg.EARLY_STOP_WORST_SUCCESS
            ):
                torch.save(
                    warm_start_checkpoint,
                    final_model_path,
                )
                print(
                    "Perfect deterministic target and radar "
                    "validation reached."
                )
                print(
                    "PPO phase skipped to preserve the repaired "
                    "HAPPO policy."
                )
                publish_happo_dashboard_episodes(
                    happo,
                    dashboard_training_episodes,
                    training_mode,
                    completed_episodes,
                )
                return

    for episode in range(1, cfg.NUM_EPISODES + 1):
        completed_episodes = episode
        happo.set_entropy_coef(
            get_entropy_coefficient(episode)
        )

        (
            curriculum_stage,
            scout_start,
            hunter_start,
            strike_point,
            radar_layout_index,
        ) = get_training_scenario(episode)

        if episode in (
            1,
            cfg.CURRICULUM_STAGE_1_END + 1,
            cfg.CURRICULUM_STAGE_2_END + 1,
        ):
            print(
                f"Curriculum stage: {curriculum_stage} | "
                f"Entropy: {happo.entropy_coef:.5f}"
            )

        radars_config = get_curriculum_radars(
            curriculum_stage,
            radar_layout_index,
            forbidden_positions=(
                scout_start,
                hunter_start,
                strike_point,
            ),
        )

        env = TacticalEnv(
            weather=CLEAR_TRAINING_WEATHER,
            scout_start_position=scout_start,
            hunter_start_position=hunter_start,
            strike_point=strike_point,
            radars_config=radars_config,
        )

        env.reset()

        done = False
        episode_reward = 0.0
        episode_scout_reward = 0.0
        episode_hunter_reward = 0.0
        episode_steps = 0

        while not done:

            # --------------------------------------------------
            # Current CTDE observations
            # --------------------------------------------------

            scout_obs, hunter_obs, global_state = (
                get_ctde_observations(env)
            )

            # --------------------------------------------------
            # Central critic value
            # --------------------------------------------------

            global_tensor = torch.as_tensor(
                global_state,
                dtype=torch.float32,
                device=happo.device,
            ).unsqueeze(0)

            with torch.no_grad():
                value = happo.critic(
                    global_tensor
                ).squeeze().item()

            # --------------------------------------------------
            # Decentralized action selection
            # --------------------------------------------------

            (
                scout_action,
                hunter_action,
                scout_log_prob,
                hunter_log_prob,
            ) = happo.select_actions(
                scout_obs,
                hunter_obs,
            )

            # --------------------------------------------------
            # Environment step
            # --------------------------------------------------

            _, reward, done, info = env.step(
                scout_action=scout_action,
                hunter_action=hunter_action,
            )

            scout_reward = float(
                info.get("reward_scout", reward)
            )

            hunter_reward = float(
                info.get("reward_hunter", reward)
            )

            # --------------------------------------------------
            # Store transition
            # --------------------------------------------------

            buffer.add(
                scout_obs=scout_obs,
                hunter_obs=hunter_obs,
                global_state=global_state,
                scout_action=scout_action,
                hunter_action=hunter_action,
                scout_log_prob=scout_log_prob,
                hunter_log_prob=hunter_log_prob,
                reward=reward,
                scout_reward=scout_reward,
                hunter_reward=hunter_reward,
                value=value,
                done=done,
            )

            episode_reward += float(reward)
            episode_scout_reward += scout_reward
            episode_hunter_reward += hunter_reward

            episode_steps += 1

            # --------------------------------------------------
            # Update after rollout
            # --------------------------------------------------

            if len(buffer) >= cfg.ROLLOUT_LENGTH:

                next_value = 0.0

                if not done:
                    _, _, next_global_state = (
                        get_ctde_observations(env)
                    )

                    next_global_tensor = torch.as_tensor(
                        next_global_state,
                        dtype=torch.float32,
                        device=happo.device,
                    ).unsqueeze(0)

                    with torch.no_grad():
                        next_value = (
                            happo.critic(next_global_tensor)
                            .squeeze()
                            .item()
                        )

                metrics = happo.update(
                    buffer.get(),
                    next_value=next_value,
                )
                latest_update_metrics = metrics
                happo_update_count += 1

                if (
                    scratch_demonstrations is not None
                    and happo_update_count
                    % cfg.SCRATCH_GUIDANCE_UPDATE_INTERVAL
                    == 0
                ):
                    train_actors_from_demonstrations(
                        happo,
                        scratch_demonstrations,
                        epochs=(
                            cfg.SCRATCH_GUIDANCE_REFRESH_EPOCHS
                        ),
                    )

                buffer.clear()

        # ------------------------------------------------------
        # Episode statistics
        # ------------------------------------------------------

        episode_rewards.append(episode_reward)

        episode_successes.append(
            bool(info.get("mission_success", False))
        )
        episode_scout_rewards.append(
            episode_scout_reward
        )
        episode_hunter_rewards.append(
            episode_hunter_reward
        )
        episode_step_counts.append(episode_steps)

        if (
            episode == 1
            or episode % cfg.PRINT_INTERVAL == 0
        ):
            dashboard_training_episodes.append(
                {
                    "label": f"HAPPO Training Episode {episode}",
                    "steps": deepcopy(env.episode_log),
                }
            )

        if episode % cfg.PRINT_INTERVAL == 0:
            interval_slice = slice(-cfg.PRINT_INTERVAL, None)
            average_reward = np.mean(
                episode_rewards[interval_slice]
            )
            success_rate = (
                np.mean(
                    episode_successes[interval_slice]
                )
                * 100.0
            )
            average_scout_reward = np.mean(
                episode_scout_rewards[interval_slice]
            )
            average_hunter_reward = np.mean(
                episode_hunter_rewards[interval_slice]
            )
            average_steps = np.mean(
                episode_step_counts[interval_slice]
            )

            loss_summary = ""

            if latest_update_metrics is not None:
                loss_summary = (
                    " | Critic Loss="
                    f"{latest_update_metrics['critic_loss']:.4f}"
                    " | Scout Loss="
                    f"{latest_update_metrics['scout_policy_loss']:.4f}"
                    " | Hunter Loss="
                    f"{latest_update_metrics['hunter_policy_loss']:.4f}"
                )

            print(
                f"Episode={episode:4d} | "
                f"Average Reward={average_reward:8.2f} | "
                f"Success Rate={success_rate:6.2f}% | "
                f"Scout Reward={average_scout_reward:8.2f} | "
                f"Hunter Reward={average_hunter_reward:8.2f} | "
                f"Average Steps={average_steps:6.2f}"
                f"{loss_summary}"
            )

        # ------------------------------------------------------
        # Balanced validation and robust best-model selection
        # ------------------------------------------------------

        stop_early = False

        if episode % cfg.VALIDATION_INTERVAL == 0:
            validation = evaluate_training_targets(happo)
            current_validation_score = validation_score(
                validation
            )

            target_summary = " | ".join(
                f"{target}={result['success_rate']:.0f}%"
                for target, result
                in validation["targets"].items()
            )

            print(
                f"[VALIDATION] episode={episode} | "
                f"Success Rate={validation['mean_success_rate']:.2f}% | "
                f"Worst Success Rate={validation['worst_success_rate']:.2f}% | "
                f"Radar Success Rate={validation['radar_success_rate']:.2f}% | "
                f"Worst Radar={validation['worst_radar_success_rate']:.2f}% | "
                f"reward={validation['average_reward']:.2f} | "
                f"{target_summary}"
            )

            if (
                episode >= cfg.BEST_MODEL_START_EPISODE
                and current_validation_score
                > best_validation_score
            ):
                best_validation_score = current_validation_score
                best_worst_success_rate = float(
                    validation["worst_success_rate"]
                )
                best_success_rate = float(
                    validation["mean_success_rate"]
                )
                best_radar_success_rate = float(
                    validation["radar_success_rate"]
                )
                best_worst_radar_success_rate = float(
                    validation[
                        "worst_radar_success_rate"
                    ]
                )
                best_reward = float(
                    validation["average_reward"]
                )

                torch.save(
                    {
                        "episode": episode,
                        "scout_actor": (
                            happo.scout_actor.state_dict()
                        ),
                        "hunter_actor": (
                            happo.hunter_actor.state_dict()
                        ),
                        "critic": (
                            happo.critic.state_dict()
                        ),
                        "best_worst_success_rate": (
                            best_worst_success_rate
                        ),
                        "best_success_rate": best_success_rate,
                        "best_radar_success_rate": (
                            best_radar_success_rate
                        ),
                        "best_worst_radar_success_rate": (
                            best_worst_radar_success_rate
                        ),
                        "best_average_reward": best_reward,
                        "validation_targets": validation["targets"],
                        "validation_radar_layouts": (
                            validation["radar_layouts"]
                        ),
                    },
                    best_model_path,
                )

                print(
                    "-> Best HAPPO model saved "
                    f"(Success Rate={best_success_rate:.2f}% | "
                    f"Worst Success Rate={best_worst_success_rate:.2f}% | "
                    f"Radar Success Rate={best_radar_success_rate:.2f}% | "
                    f"Worst Radar={best_worst_radar_success_rate:.2f}% | "
                    f"avg={best_reward:.2f})"
                )

            full_curriculum_reached = (
                resume_best
                or episode > cfg.CURRICULUM_STAGE_2_END
            )

            if (
                full_curriculum_reached
                and validation["worst_success_rate"]
                >= cfg.EARLY_STOP_WORST_SUCCESS
                and validation["worst_radar_success_rate"]
                >= cfg.EARLY_STOP_WORST_SUCCESS
            ):
                perfect_validation_streak += 1
            else:
                perfect_validation_streak = 0

            if (
                episode >= cfg.BEST_MODEL_START_EPISODE
                and perfect_validation_streak
                >= cfg.EARLY_STOP_PATIENCE
            ):
                stop_early = True
                print(
                    "Early stopping: all targets and radar layouts "
                    f"reached {cfg.EARLY_STOP_WORST_SUCCESS:.0f}% "
                    f"for {perfect_validation_streak} validations."
                )

        # ------------------------------------------------------
        # Periodic checkpoint
        # ------------------------------------------------------

        if episode % 100 == 0:

            torch.save(
                {
                    "episode": episode,
                    "scout_actor": (
                        happo.scout_actor.state_dict()
                    ),
                    "hunter_actor": (
                        happo.hunter_actor.state_dict()
                    ),
                    "critic": (
                        happo.critic.state_dict()
                    ),
                    "episode_rewards": episode_rewards,
                    "episode_successes": episode_successes,
                },
                Path("models")
                / f"{checkpoint_prefix}_{episode}.pth",
            )

            print(
                f"  -> Checkpoint saved: "
                f"{checkpoint_prefix}_{episode}.pth"
            )

        if stop_early:
            break

    # ----------------------------------------------------------
    # Final rollout update
    # ----------------------------------------------------------

    if len(buffer) > 0:

        metrics = happo.update(
            buffer.get()
        )

        print(
            f"[FINAL UPDATE] "
            f"critic={metrics['critic_loss']:.5f} | "
            f"scout={metrics['scout_policy_loss']:.5f} | "
            f"hunter={metrics['hunter_policy_loss']:.5f}"
        )

        buffer.clear()

    # ----------------------------------------------------------
    # Restore and save the best validated model
    # ----------------------------------------------------------

    if best_model_path.exists():
        best_checkpoint = torch.load(
            best_model_path,
            map_location=happo.device,
            weights_only=True,
        )
        happo.scout_actor.load_state_dict(
            best_checkpoint["scout_actor"]
        )
        happo.hunter_actor.load_state_dict(
            best_checkpoint["hunter_actor"]
        )

        if "critic" in best_checkpoint:
            happo.critic.load_state_dict(
                best_checkpoint["critic"]
            )

        print("Best validated policy restored for final save.")

    torch.save(
        {
            "episode": completed_episodes,
            "training_phase": "happo_training_complete",
            "training_mode": training_mode,
            "scout_actor": (
                happo.scout_actor.state_dict()
            ),
            "hunter_actor": (
                happo.hunter_actor.state_dict()
            ),
            "critic": (
                happo.critic.state_dict()
            ),
            "episode_rewards": episode_rewards,
            "episode_successes": episode_successes,
            "best_average_reward": best_reward,
            "best_success_rate": best_success_rate,
            "best_worst_success_rate": best_worst_success_rate,
            "best_radar_success_rate": (
                best_radar_success_rate
            ),
            "best_worst_radar_success_rate": (
                best_worst_radar_success_rate
            ),
        },
        final_model_path,
    )

    publish_happo_dashboard_episodes(
        happo,
        dashboard_training_episodes,
        training_mode,
        completed_episodes,
    )

    print()
    print("=" * 60)
    print("HAPPO TRAINING COMPLETE")
    print("=" * 60)
    print(
        f"Best Success Rate: "
        f"{best_success_rate:.2f}%"
    )
    print(
        f"Best worst-target: "
        f"{best_worst_success_rate:.2f}%"
    )
    print(
        f"Best worst-radar : "
        f"{best_worst_radar_success_rate:.2f}%"
    )
    print(
        f"Best avg{cfg.BEST_MODEL_WINDOW}: "
        f"{best_reward:.2f}"
    )
    print(
        f"Final avg20: "
        f"{np.mean(episode_rewards[-20:]):.2f}"
    )
    print("Saved:")
    print(f"  {best_model_path}")
    print(f"  {final_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the HAPPO escort policy.",
    )
    parser.add_argument(
        "--mode",
        choices=("scratch", "resume"),
        default="scratch",
        help=(
            "scratch starts from random weights and writes separate "
            "checkpoints; resume repairs and continues happo_best.pth"
        ),
    )
    arguments = parser.parse_args()
    main(arguments.mode)
