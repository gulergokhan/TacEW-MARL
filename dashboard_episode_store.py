"""Shared persistence helpers for Sortie Playback episode logs."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
JSON_PATH = ROOT / "dashboard_logs" / "tactical_episodes.json"
JS_PATH = ROOT / "dashboard_logs" / "tactical_episodes.js"


def load_dashboard_episodes():
    """Load the dashboard episode list from JSON or its JS fallback."""

    source_path = JSON_PATH if JSON_PATH.exists() else JS_PATH

    if not source_path.exists():
        return []

    try:
        text = source_path.read_text(encoding="utf-8")

        if source_path.suffix == ".js":
            _, separator, json_text = text.partition("=")

            if not separator:
                return []

            text = json_text.strip().rstrip(";")

        payload = json.loads(text)
        episodes = payload.get("episodes", [])
        return episodes if isinstance(episodes, list) else []
    except (OSError, ValueError, json.JSONDecodeError):
        return []


def save_dashboard_episodes(episodes):
    """Write matching JSON and browser-loadable JS episode files."""

    payload = {"episodes": episodes}
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    JS_PATH.write_text(
        "window.TACEW_EPISODES = "
        + json.dumps(payload)
        + ";\n",
        encoding="utf-8",
    )


def replace_episode_group(new_episodes, label_prefixes):
    """Replace one producer's episodes while preserving other groups."""

    prefixes = tuple(label_prefixes)
    existing = [
        episode
        for episode in load_dashboard_episodes()
        if not str(episode.get("label", "")).startswith(prefixes)
    ]
    merged = existing + list(new_episodes)
    save_dashboard_episodes(merged)
    return merged
