"""
Runs your custom Scenario Builder map through TacticalEnv and writes the
result as a playable episode into dashboard_logs/tactical_episodes.js.

scenario.json (downloaded from the dashboard's Scenario Builder tab) is a
MAP DEFINITION, not an episode log — it has no per-step positions, radar
states or rewards, so the dashboard's "LOAD JSON" button (which expects an
episode log) can't render it directly. This script is the missing link:
it builds the env from your scenario, runs one sortie through it with a
simple heuristic Scout policy, and exports that sortie in the format the
Sortie Playback map understands.

Usage:
    python3 run_scenario.py path/to/scenario.json

    # or, if scenario.json already sits in the project root:
    python3 run_scenario.py

After it finishes, just refresh tacew_dashboard.html in the browser (no
need to use the "LOAD JSON" button) — your scenario appears as a new
entry in the episode dropdown, on the real map, with your terrain and
radar layout.
"""

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import torch

from dashboard_episode_store import (
    load_dashboard_episodes,
    save_dashboard_episodes,
)
from evaluate_happo import (
    get_ctde_observations as get_happo_observations,
    load_best_model as load_best_happo_model,
    resolve_happo_model_path,
    select_deterministic_actions as select_happo_actions,
)
from marl.algorithms.happo import HAPPO
from marl.execution import select_guarded_dqn_actions
from environment.tactical_env import TacticalEnv
from evaluate_tactical_dqn import (
    DEFAULT_MODEL_PATH,
    load_agent,
)

MAX_CUSTOM_DASHBOARD_EPISODES = 20
CUSTOM_SCENARIO_POLICY_PREFIXES = (
    "Custom scenario (trained dqn",
    "Custom scenario (happo",
    "Custom scenario (heuristic",
    "Custom scenario (dqn",
)
CUSTOM_MODEL_PATH = (
    Path(__file__).resolve().parent / "models" / "custom_scenario_dqn.pth"
)


def keep_latest_custom_scenario_per_policy(episodes):
    """Keep only the latest playback for each known policy type."""

    seen_policies = set()
    retained_reversed = []

    for episode in reversed(episodes):
        label = str(episode.get("label", ""))
        policy_prefix = next(
            (
                prefix
                for prefix in CUSTOM_SCENARIO_POLICY_PREFIXES
                if label.startswith(prefix)
            ),
            None,
        )

        if policy_prefix is not None:
            if policy_prefix in seen_policies:
                continue

            seen_policies.add(policy_prefix)

        retained_reversed.append(episode)

    return list(reversed(retained_reversed))


def scripted_scout_policy(env, rng):
    """Same simple heuristic as generate_dashboard_demo.py: move toward the
    strike point, jam the nearest radar when one is close enough to matter.
    Swap this out for your own trained policy's action selection once you
    have one."""
    sx, sy = env.scout.state.position.x, env.scout.state.position.y
    tx, ty = env.strike_point.x, env.strike_point.y

    nearest = env.radar_system.nearest_radar(env.scout.state.position, max_range=2.0)
    if nearest is not None and rng.random() < 0.6:
        return env.ACTION_JAM_SUPPRESS if rng.random() < 0.5 else env.ACTION_JAM_DECEIVE

    if rng.random() < 0.15:
        return env.ACTION_STAY

    if abs(tx - sx) > abs(ty - sy):
        return env.ACTION_RIGHT if tx > sx else env.ACTION_LEFT
    if ty != sy:
        return env.ACTION_DOWN if ty > sy else env.ACTION_UP
    return env.ACTION_STAY


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a dashboard scenario with " "the trained DQN or heuristic policy."
        )
    )

    parser.add_argument(
        "scenario",
        nargs="?",
        type=Path,
        default=Path("scenario.json"),
    )

    parser.add_argument(
        "--policy",
        choices=("dqn","happo", "heuristic"),
        default="dqn",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=0,
        help=("Train repeatedly on this scenario " "before the final rollout."),
    )

    parser.add_argument(
        "--output-model",
        type=Path,
        default=CUSTOM_MODEL_PATH,
    )

    return parser.parse_args()


def greedy_rollout(agent, env):
    state = env.reset()
    done = False
    total_reward = 0.0
    info = {}

    while not done:
        action = agent.select_action(
            state,
            training=False,
        )

        state, reward, done, info = env.step(
            action,
            hunter_action=None,
        )

        total_reward += reward

    return {
        "reward": total_reward,
        "success": bool(info.get("mission_success", False)),
    }
def guided_training_action(env):
    scout_position = env.scout.state.position
    hunter_position = env.hunter.state.position

    nearest_radar = (
        env.radar_system.nearest_radar(
            scout_position,
            max_range=3.0,
        )
    )

    if nearest_radar is not None:
        hunter_radar_distance = (
            env.radar_system.distance(
                nearest_radar.position,
                hunter_position,
            )
        )

        hunter_approaching_radar = (
            hunter_radar_distance
            <= nearest_radar.detection_range + 1.0
        )

        if (
            hunter_approaching_radar
            and nearest_radar.suppression_timer <= 0
        ):
            return env.ACTION_JAM_SUPPRESS

    scout_x = scout_position.x
    scout_y = scout_position.y
    hunter_x = hunter_position.x
    hunter_y = hunter_position.y

    candidates = []

    if hunter_x > scout_x:
        candidates.append(env.ACTION_RIGHT)
    elif hunter_x < scout_x:
        candidates.append(env.ACTION_LEFT)

    if hunter_y > scout_y:
        candidates.append(env.ACTION_DOWN)
    elif hunter_y < scout_y:
        candidates.append(env.ACTION_UP)

    if candidates:
        return random.choice(candidates)

    return env.ACTION_STAY

def train_on_scenario(agent, env, episodes):
    agent.model.train()
    agent.epsilon = 1.0
    agent.epsilon_min = 0.05

    agent.epsilon_decay = agent.epsilon_min ** (1.0 / episodes)

    initial_evaluation = greedy_rollout(
        agent,
        env,
    )

    best_score = (
        int(initial_evaluation["success"]),
        initial_evaluation["reward"],
    )

    best_model_state = deepcopy(agent.model.state_dict())

    print(
        f"Initial evaluation | "
        f"Reward={initial_evaluation['reward']:.2f} | "
        f"Success={initial_evaluation['success']}",
        flush=True,
    )

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        losses = []

        while not done:
            if random.random() < agent.epsilon:
                if random.random() < 0.8:
                    action = guided_training_action(
                        env
                    )
                else:
                    action = random.randrange(
                        agent.action_size
                    )
            else:
                action = agent.select_action(
                    state,
                    training=False,
                )

            next_state, reward, done, _ = env.step(
                action,
                hunter_action=None,
            )

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
            )

            loss = agent.learn()

            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        should_evaluate = episode == 1 or episode % 25 == 0 or episode == episodes

        if not should_evaluate:
            continue

        evaluation = greedy_rollout(
            agent,
            env,
        )

        score = (
            int(evaluation["success"]),
            evaluation["reward"],
        )

        if score > best_score:
            best_score = score
            best_model_state = deepcopy(agent.model.state_dict())

        average_loss = sum(losses) / len(losses) if losses else 0.0

        print(
            f"Episode={episode:4d} | "
            f"Train Reward={total_reward:8.2f} | "
            f"Greedy Reward={evaluation['reward']:8.2f} | "
            f"Success={evaluation['success']} | "
            f"Loss={average_loss:.4f} | "
            f"Epsilon={agent.epsilon:.3f}",
            flush=True,
        )

    agent.model.load_state_dict(best_model_state)
    agent.target_model.load_state_dict(best_model_state)

    agent.model.eval()
    agent.epsilon = 0.0


def main():
    args = parse_args()
    scenario_path = args.scenario

    if not scenario_path.exists():
        print(f"Can't find {scenario_path}.")
        print("Pass the path explicitly, e.g.:")
        print("    python3 run_scenario.py ~/Downloads/scenario.json")
        raise SystemExit(1)

    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))

    print(
        f"Running scenario: {scenario_path.name} | " f"Policy: {args.policy}",
        flush=True,
    )

    env = TacticalEnv.from_scenario(scenario)

    dqn_agent = None
    happo_agent = None
    happo_model_path = None

    if args.train_episodes < 0:
        raise ValueError("--train-episodes cannot be negative.")

    if args.train_episodes > 0 and args.policy != "dqn":
        raise ValueError("Scenario training requires the dqn policy.")

    if args.policy == "dqn":
        dqn_agent = load_agent(
            env,
            args.model,
        )

    elif args.policy == "happo":
        happo_agent = HAPPO()
        happo_model_path = resolve_happo_model_path()
        load_best_happo_model(
            happo_agent,
            model_path=happo_model_path,
        )
        print(
            f"HAPPO model: {happo_model_path}",
            flush=True,
        )

    if args.train_episodes > 0:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        print(
            f"Training on custom scenario for " f"{args.train_episodes} episodes...",
            flush=True,
        )

        train_on_scenario(
            dqn_agent,
            env,
            args.train_episodes,
        )

        args.output_model.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            dqn_agent.model.state_dict(),
            args.output_model,
        )

        print(
            f"Custom model saved to: " f"{args.output_model}",
            flush=True,
        )

    observation = env.reset()
    rng = random.Random(args.seed)
    done = False

    while not done:
        hunter_action = None

        if dqn_agent is not None:
            scout_action = dqn_agent.select_action(
                observation,
                training=False,
            )
            scout_action, hunter_action = (
                select_guarded_dqn_actions(
                    env,
                    scout_action,
                )
            )

        elif happo_agent is not None:
            scout_obs, hunter_obs = (
                get_happo_observations(env)
            )

            scout_action, hunter_action = (
                select_happo_actions(
                    happo_agent,
                    scout_obs,
                    hunter_obs,
                    env=env,
                )
            )

        else:
            scout_action = scripted_scout_policy(
                env,
                rng,
            )

        observation, _, done, _ = env.step(
            scout_action=scout_action,
            hunter_action=hunter_action,
        )

    policy_label = "trained dqn" if args.train_episodes > 0 else args.policy

    new_episode = {
        "label": (
            f"Custom scenario ({policy_label}"
            + (
                " + tactical guard"
                if args.policy in {"dqn", "happo"}
                else ""
            )
            + (
                f" · {happo_model_path.stem}"
                if happo_model_path is not None
                else ""
            )
            + ") — "
            f"{scenario_path.name}"
        ),
        "steps": env.episode_log,
    }
    existing_episodes = load_dashboard_episodes()

    training_episodes = [
        episode
        for episode in existing_episodes
        if not str(episode.get("label", "")).startswith("Custom scenario")
    ]

    custom_episodes = [
        episode
        for episode in existing_episodes
        if str(episode.get("label", "")).startswith("Custom scenario")
    ]

    custom_episodes.append(new_episode)
    custom_episodes = (
        keep_latest_custom_scenario_per_policy(
            custom_episodes
        )
    )

    existing_episodes = (
        training_episodes + custom_episodes[-MAX_CUSTOM_DASHBOARD_EPISODES:]
    )

    save_dashboard_episodes(existing_episodes)

    outcome = new_episode["steps"][-1].get("termination_reason")

    print(
        f"Ran your scenario: "
        f"{len(new_episode['steps'])} steps, "
        f"outcome={outcome}"
    )

    print(f"Dashboard now has " f"{len(existing_episodes)} episode(s).")

    print("The new episode is ready in " "Sortie Playback.")


if __name__ == "__main__":
    main()
