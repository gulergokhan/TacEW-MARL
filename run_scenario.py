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

import json
import random
import sys
from pathlib import Path

from environment.tactical_env import TacticalEnv


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


def main():
    scenario_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scenario.json")

    if not scenario_path.exists():
        print(f"Can't find {scenario_path}.")
        print("Pass the path explicitly, e.g.:")
        print("    python3 run_scenario.py ~/Downloads/scenario.json")
        sys.exit(1)

    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))

    env = TacticalEnv.from_scenario(scenario)
    env.reset()

    rng = random.Random(0)
    done = False
    while not done:
        scout_action = scripted_scout_policy(env, rng)
        _, _, done, _ = env.step(scout_action, hunter_action=None)

    new_episode = {
        "label": f"Custom scenario — {scenario_path.name}",
        "steps": env.episode_log,
    }

    output_path = Path("dashboard_logs/tactical_episodes.js")

    existing_episodes = []
    if output_path.exists():
        text = output_path.read_text(encoding="utf-8")
        # strip the "window.TACEW_EPISODES = " prefix and trailing ";"
        json_text = text.split("=", 1)[1].strip().rstrip(";")
        try:
            existing_episodes = json.loads(json_text).get("episodes", [])
        except (json.JSONDecodeError, IndexError):
            existing_episodes = []

    existing_episodes.append(new_episode)
    payload = {"episodes": existing_episodes}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("window.TACEW_EPISODES = ")
        json.dump(payload, f)
        f.write(";\n")

    outcome = new_episode["steps"][-1].get("termination_reason")
    print(f"Ran your scenario: {len(new_episode['steps'])} steps, outcome={outcome}")
    print(f"Appended to {output_path} — now has {len(existing_episodes)} episode(s).")
    print("Refresh tacew_dashboard.html in the browser and pick it from the episode dropdown.")


if __name__ == "__main__":
    main()
