import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dashboard_episode_store as store


class TestDashboardEpisodeStore(unittest.TestCase):

    def test_save_and_load_keep_json_and_js_in_sync(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "episodes.json"
            js_path = root / "episodes.js"
            episodes = [{"label": "Example", "steps": []}]

            with (
                patch.object(store, "JSON_PATH", json_path),
                patch.object(store, "JS_PATH", js_path),
            ):
                store.save_dashboard_episodes(episodes)

                self.assertEqual(
                    store.load_dashboard_episodes(),
                    episodes,
                )
                js_payload = js_path.read_text(encoding="utf-8")
                self.assertTrue(
                    js_payload.startswith("window.TACEW_EPISODES = ")
                )

    def test_replace_group_preserves_other_episode_producers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "episodes.json"
            js_path = root / "episodes.js"
            json_path.write_text(
                json.dumps(
                    {
                        "episodes": [
                            {"label": "DQN Training Episode 50"},
                            {"label": "HAPPO Training Episode 50"},
                            {"label": "Custom scenario (dqn)"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch.object(store, "JSON_PATH", json_path),
                patch.object(store, "JS_PATH", js_path),
            ):
                merged = store.replace_episode_group(
                    [{"label": "HAPPO Final Policy"}],
                    (
                        "HAPPO Training Episode",
                        "HAPPO Final Policy",
                    ),
                )

            labels = [episode["label"] for episode in merged]
            self.assertEqual(
                labels,
                [
                    "DQN Training Episode 50",
                    "Custom scenario (dqn)",
                    "HAPPO Final Policy",
                ],
            )


if __name__ == "__main__":
    unittest.main()
