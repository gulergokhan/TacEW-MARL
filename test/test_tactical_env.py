import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from environment.tactical_env import TacticalEnv
from weather.models import WeatherState


def clear_weather():
    return WeatherState(
        temperature=20.0,
        wind_speed=0.0,
        wind_direction=0.0,
        visibility=10000.0,
        precipitation=0.0,
        cloud_cover=0.0,
    )


class TestTacticalEnv(unittest.TestCase):

    def setUp(self):
        self.env = TacticalEnv(weather=clear_weather())
        self.observation = self.env.reset()

    def test_reset_returns_expected_observation_and_action_sizes(self):
        self.assertEqual(len(self.observation), 99)
        self.assertEqual(self.env.NUM_ACTIONS, 7)

    def test_hunter_heuristic_moves_toward_strike_point(self):
        self.env.step(self.env.ACTION_STAY)

        self.assertEqual(
            (
                self.env.hunter.state.position.x,
                self.env.hunter.state.position.y,
            ),
            (7, 8),
        )

    def test_suppression_jam_targets_nearest_radar(self):
        self.env.scout.move(4, 5)

        _, _, _, info = self.env.step(self.env.ACTION_JAM_SUPPRESS)
        radar = self.env.radar_system.get_radar("radar_01")

        self.assertGreater(radar.suppression_jam, 0.0)
        self.assertGreater(radar.suppression_timer, 0)
        self.assertEqual(info["action"], "JAM_SUPPRESS")

    def test_episode_log_can_be_exported(self):
        self.env.step(self.env.ACTION_RIGHT)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "episode.json"
            self.env.export_log(output)
            data = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["step"], 1)
        self.assertIn("terrain_grid", data[0])

    def test_dashboard_status_contains_scout_and_hunter_tracks(self):
        self.env.hunter.move(3, 8)

        with patch("radar.radar.random.random", return_value=0.0):
            _, _, _, info = self.env.step(self.env.ACTION_STAY)

        radar_02_status = [
            status
            for status in info["radar_status"]
            if status["radar_id"] == "radar_02"
        ]

        self.assertEqual(
            {status["target_id"] for status in radar_02_status},
            {"scout_01", "hunter_01"},
        )
        hunter_status = next(
            status
            for status in radar_02_status
            if status["target_id"] == "hunter_01"
        )
        self.assertEqual(hunter_status["state"], "LOCK")

    def test_escort_mission_succeeds_when_hunter_reaches_target(self):
        self.env.scout.move(2, 8)
        self.env.hunter.move(2, 8)

        with patch(
            "radar.radar.random.random",
            return_value=1.0,
        ):
            _, reward, done, info = self.env.step(
                self.env.ACTION_STAY,
                hunter_action=self.env.ACTION_LEFT,
            )

        self.assertTrue(done)
        self.assertGreater(reward, 0.0)
        self.assertTrue(
            info["hunter"]["target_reached"]
        )
        self.assertTrue(
            info["escort_in_range"]
        )
        self.assertTrue(
            info["mission_success"]
        )
        self.assertFalse(
            info["mission_failed"]
        )
        self.assertEqual(
            info["termination_reason"],
            "mission_success",
        )

    def test_hunter_lethal_ends_mission_as_failure(self):
        self.env.hunter.move(2, 8)

        radar = self.env.radar_system.get_radar("radar_02")
        radar.lethal_lock_threshold = 1

        with patch("radar.radar.random.random", return_value=0.0):
            _, reward, done, info = self.env.step(
                self.env.ACTION_STAY,
                hunter_action=self.env.ACTION_STAY,
            )

        self.assertTrue(done)
        self.assertLess(reward, 0.0)
        self.assertTrue(info["hunter_lethal_hit"])
        self.assertFalse(info["mission_success"])
        self.assertTrue(info["mission_failed"])
        self.assertEqual(
            info["termination_reason"],
            "hunter_lethal",
        )

if __name__ == "__main__":
    unittest.main()
