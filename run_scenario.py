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
from pathlib import Path

from environment.tactical_env import TacticalEnv
from evaluate_tactical_dqn import (
    DEFAULT_MODEL_PATH,
    load_agent,
)

MAX_CUSTOM_DASHBOARD_EPISODES = 20

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
            "Run a dashboard scenario with "
            "the trained DQN or heuristic policy."
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
        choices=("dqn", "heuristic"),
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

    return parser.parse_args()

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
        f"Running scenario: {scenario_path.name} | "
        f"Policy: {args.policy}",
        flush=True,
    )

    env = TacticalEnv.from_scenario(scenario)

    agent = None

    if args.policy == "dqn":
        agent = load_agent(
            env,
            args.model,
        )

    observation = env.reset()
    rng = random.Random(args.seed)
    done = False

    while not done:
        if agent is not None:
            scout_action = agent.select_action(
                observation,
                training=False,
            )
        else:
            scout_action = scripted_scout_policy(
                env,
                rng,
            )

        observation, _, done, _ = env.step(
            scout_action,
            hunter_action=None,
        )

    new_episode = {
        "label": (
            f"Custom scenario ({args.policy}) — "
            f"{scenario_path.name}"
        ),
        "steps": env.episode_log,
    }
    json_output_path = Path(
        "dashboard_logs/tactical_episodes.json"
    )
    js_output_path = Path(
        "dashboard_logs/tactical_episodes.js"
    )

    existing_episodes = []

    source_path = (
        json_output_path
        if json_output_path.exists()
        else js_output_path
    )

    if source_path.exists():
        try:
            text = source_path.read_text(
                encoding="utf-8"
            )

            if source_path.suffix == ".js":
                _, separator, json_text = (
                    text.partition("=")
                )

                if not separator:
                    raise ValueError(
                        "Invalid dashboard episode file"
                    )

                text = json_text.strip().rstrip(";")

            existing_payload = json.loads(text)
            existing_episodes = (
                existing_payload.get("episodes", [])
            )

            if not isinstance(
                existing_episodes,
                list,
            ):
                raise ValueError(
                    "Dashboard episodes must be a list"
                )

        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
        ):
            existing_episodes = []

    training_episodes = [
        episode
        for episode in existing_episodes
        if not str(
            episode.get("label", "")
        ).startswith("Custom scenario")
    ]

    custom_episodes = [
        episode
        for episode in existing_episodes
        if str(
            episode.get("label", "")
        ).startswith("Custom scenario")
    ]

    custom_episodes.append(new_episode)

    existing_episodes = (
        training_episodes
        + custom_episodes[
            -MAX_CUSTOM_DASHBOARD_EPISODES:
        ]
    )

    payload = {
        "episodes": existing_episodes,
    }

    json_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    js_output_path.write_text(
        "window.TACEW_EPISODES = "
        + json.dumps(payload)
        + ";\n",
        encoding="utf-8",
    )

    outcome = new_episode["steps"][-1].get(
        "termination_reason"
    )

    print(
        f"Ran your scenario: "
        f"{len(new_episode['steps'])} steps, "
        f"outcome={outcome}"
    )

    print(
        f"Dashboard now has "
        f"{len(existing_episodes)} episode(s)."
    )

    print(
        "The new episode is ready in "
        "Sortie Playback."
    )


if __name__ == "__main__":
    main()
