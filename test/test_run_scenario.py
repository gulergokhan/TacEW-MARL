import unittest

from run_scenario import (
    keep_latest_custom_scenario_per_policy,
)


class TestRunScenario(unittest.TestCase):

    def test_keeps_latest_episode_for_each_policy(self):
        episodes = [
            {"label": "Custom scenario (dqn) — old"},
            {"label": "Custom scenario (happo) — old"},
            {"label": "Custom scenario (heuristic) — map"},
            {"label": "Custom scenario (trained dqn) — map"},
            {"label": "Custom scenario (dqn) — new"},
            {"label": "Custom scenario (happo) — new"},
        ]

        compacted = keep_latest_custom_scenario_per_policy(
            episodes
        )

        self.assertEqual(
            [episode["label"] for episode in compacted],
            [
                "Custom scenario (heuristic) — map",
                "Custom scenario (trained dqn) — map",
                "Custom scenario (dqn) — new",
                "Custom scenario (happo) — new",
            ],
        )

    def test_preserves_unknown_custom_episode_labels(self):
        episodes = [
            {"label": "Custom scenario (legacy) — one"},
            {"label": "Custom scenario (legacy) — two"},
        ]

        self.assertEqual(
            keep_latest_custom_scenario_per_policy(episodes),
            episodes,
        )


if __name__ == "__main__":
    unittest.main()
