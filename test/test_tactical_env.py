import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from terrain.models import TerrainType

from configs import environment_config as cfg
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

    def test_custom_escort_start_positions(self):
        env = TacticalEnv(
            weather=clear_weather(),
            scout_start_position=(5, 6),
            hunter_start_position=(6, 6),
        )

        env.reset()

        self.assertEqual(
            (
                env.scout.state.position.x,
                env.scout.state.position.y,
            ),
            (5, 6),
        )

        self.assertEqual(
            (
                env.hunter.state.position.x,
                env.hunter.state.position.y,
            ),
            (6, 6),
        )

    def setUp(self):
        self.env = TacticalEnv(weather=clear_weather())
        self.observation = self.env.reset()

    def test_reset_returns_expected_observation_and_action_sizes(self):
        self.assertEqual(len(self.observation), 101)
        self.assertEqual(self.env.NUM_ACTIONS, 7)

    def test_strike_point_changes_agent_observations(self):
        first_env = TacticalEnv(
            weather=clear_weather(),
            strike_point=(1, 8),
        )
        second_env = TacticalEnv(
            weather=clear_weather(),
            strike_point=(8, 1),
        )

        first_observation = first_env.reset()
        second_observation = second_env.reset()

        self.assertNotEqual(
            first_observation[6:8].tolist(),
            second_observation[6:8].tolist(),
        )

        with patch(
            "radar.radar.random.random",
            return_value=1.0,
        ):
            _, _, _, first_info = first_env.step(
                first_env.ACTION_STAY,
                hunter_action=first_env.ACTION_STAY,
            )
            _, _, _, second_info = second_env.step(
                second_env.ACTION_STAY,
                hunter_action=second_env.ACTION_STAY,
            )

        self.assertNotEqual(
            first_info["observations"]["scout"][7:9].tolist(),
            second_info["observations"]["scout"][7:9].tolist(),
        )
        self.assertNotEqual(
            first_info["observations"]["hunter"][7:9].tolist(),
            second_info["observations"]["hunter"][7:9].tolist(),
        )

    def test_hunter_heuristic_moves_toward_strike_point(self):
        self.env.step(self.env.ACTION_STAY)

        self.assertEqual(
            (
                self.env.hunter.state.position.x,
                self.env.hunter.state.position.y,
            ),
            (7, 8),
        )
    def test_hunter_avoids_active_radar_cell(self):
        self.env.hunter.move(2, 6)

        self.env._hunter_heuristic_move()

        hunter_position = (
            self.env.hunter.state.position.x,
            self.env.hunter.state.position.y,
        )

        self.assertEqual(
            hunter_position,
            (1, 6),
        )

        self.assertNotEqual(
            hunter_position,
            (2, 7),
        )

    def test_suppression_jam_targets_nearest_radar(self):
        self.env.scout.move(4, 5)

        _, _, _, info = self.env.step(self.env.ACTION_JAM_SUPPRESS)
        radar = self.env.radar_system.get_radar("radar_01")

        self.assertGreater(radar.suppression_jam, 0.0)
        self.assertGreater(radar.suppression_timer, 0)
        self.assertEqual(info["action"], "JAM_SUPPRESS")

    def test_repeated_suppression_jam_does_not_refresh_timer(self):
        self.env.scout.move(4, 5)

        with patch(
            "radar.radar.random.random",
            return_value=1.0,
        ):
            self.env.step(
                self.env.ACTION_JAM_SUPPRESS,
                hunter_action=self.env.ACTION_STAY,
            )
            radar = self.env.radar_system.get_radar("radar_01")
            first_timer = radar.suppression_timer

            self.env.step(
                self.env.ACTION_JAM_SUPPRESS,
                hunter_action=self.env.ACTION_STAY,
            )

        self.assertEqual(
            radar.suppression_timer,
            first_timer - 1,
        )

    def test_suppression_moves_to_next_available_radar(self):
        self.env.scout.move(4, 6)
        first_radar = self.env.radar_system.get_radar(
            "radar_01"
        )
        second_radar = self.env.radar_system.get_radar(
            "radar_02"
        )

        with patch(
            "radar.radar.random.random",
            return_value=1.0,
        ):
            self.env.step(
                self.env.ACTION_JAM_SUPPRESS,
                hunter_action=self.env.ACTION_STAY,
            )
            self.env.step(
                self.env.ACTION_JAM_SUPPRESS,
                hunter_action=self.env.ACTION_STAY,
            )

        self.assertGreater(
            first_radar.suppression_timer,
            0,
        )
        self.assertGreater(
            second_radar.suppression_timer,
            0,
        )

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

    def test_fuel_exhaustion_has_terminal_penalty(self):
        env = TacticalEnv(
            weather=clear_weather(),
            radars_config=[],
        )
        env.reset()
        env.scout.state.fuel = 1.0
        env.hunter.state.fuel = 1.0

        _, reward, done, info = env.step(
            env.ACTION_STAY,
            hunter_action=env.ACTION_STAY,
        )

        self.assertTrue(done)
        self.assertEqual(
            info["termination_reason"],
            "fuel_exhausted",
        )
        self.assertLess(reward, 0.0)
        self.assertLessEqual(
            reward,
            cfg.FUEL_EXHAUSTED_PENALTY
            + cfg.ESCORT_REWARD,
        )

    def test_time_limit_has_terminal_penalty(self):
        env = TacticalEnv(
            weather=clear_weather(),
            radars_config=[],
        )
        env.reset()
        env.max_steps = 1

        _, reward, done, info = env.step(
            env.ACTION_STAY,
            hunter_action=env.ACTION_STAY,
        )

        self.assertTrue(done)
        self.assertEqual(
            info["termination_reason"],
            "time_limit",
        )
        self.assertLess(reward, 0.0)
        self.assertLessEqual(
            reward,
            cfg.TIME_LIMIT_PENALTY
            + cfg.ESCORT_REWARD,
        )

    def test_unescorted_hunter_target_is_penalized(self):
        self.env.scout.move(7, 7)
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
        self.assertLess(reward, 0.0)
        self.assertFalse(info["mission_success"])
        self.assertTrue(info["mission_failed"])
        self.assertFalse(info["escort_in_range"])
        self.assertEqual(
            info["termination_reason"],
            "hunter_reached_target",
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
    def test_from_scenario_loads_custom_environment(self):
        scenario = {
            "width": 12,
            "height": 8,
            "cell_km": 2.5,
            "scout_start": [1, 2],
            "hunter_start": [2, 2],
            "strike_point": [10, 6],
            "terrain_cells": [
                {
                    "x": 3,
                    "y": 3,
                    "type": "MOUNTAIN",
                }
            ],
            "radars": [
                {
                    "radar_id": "custom_01",
                    "x": 5,
                    "y": 4,
                    "detection_range": 4.5,
                }
            ],
        }
        env = TacticalEnv.from_scenario(
            scenario,
            weather=clear_weather(),
        )
        observation = env.reset()

        self.assertEqual(
            len(observation),
            101,
        )

        with patch(
            "radar.radar.random.random",
            return_value=1.0,
        ):
            _, _, _, info = env.step(
                env.ACTION_STAY,
                hunter_action=env.ACTION_STAY,
            )

        self.assertEqual(
            len(info["observations"]["scout"]),
            87,
        )
        self.assertEqual(
            len(info["observations"]["hunter"]),
            87,
        )
        self.assertEqual(
            len(info["global_state"]),
            101,
        )

        self.assertEqual(env.width, 12)
        self.assertEqual(env.height, 8)
        self.assertEqual(env.cell_km, 2.5)

        self.assertEqual(
            (
                env.scout.state.position.x,
                env.scout.state.position.y,
            ),
            (1, 2),
        )
        self.assertEqual(
            (
                env.hunter.state.position.x,
                env.hunter.state.position.y,
            ),
            (2, 2),
        )
        self.assertEqual(
            (
                env.strike_point.x,
                env.strike_point.y,
            ),
            (10, 6),
        )

        self.assertEqual(
            env.terrain.get_terrain(3, 3),
            TerrainType.MOUNTAIN,
        )

        self.assertEqual(
            len(env.radar_system.radars),
            1,
        )
        self.assertEqual(
            env.radar_system.radars[0].radar_id,
            "custom_01",
        )
    def test_observation_sizes_stay_fixed_for_supported_radar_counts(self):
        for radar_count in range(4):
            with self.subTest(
                radar_count=radar_count,
            ):
                radars = [
                    {
                        "radar_id": f"radar_{index + 1:02d}",
                        "x": index + 1,
                        "y": 4,
                        "detection_range": 3.0,
                    }
                    for index in range(radar_count)
                ]

            env = TacticalEnv.from_scenario(
                    {
                        "radars": radars,
                    },
                    weather=clear_weather(),
                )

            observation = env.reset()

            with patch(
                    "radar.radar.random.random",
                    return_value=1.0,
                ):
                    _, _, _, info = env.step(
                        env.ACTION_STAY,
                        hunter_action=env.ACTION_STAY,
                    )

            self.assertEqual(
                    len(observation),
                    101,
                )
            self.assertEqual(
                    len(info["observations"]["scout"]),
                    87,
                )
            self.assertEqual(
                    len(info["observations"]["hunter"]),
                    87,
                )
            self.assertEqual(
                    len(info["global_state"]),
                    101,
                )
    def test_scenario_rejects_invalid_terrain_cell(self):
        with self.assertRaisesRegex(
            ValueError,
            "terrain_cells",
        ):
            TacticalEnv.from_scenario(
                {
                    "width": 10,
                    "height": 10,
                    "terrain_cells": [
                        {
                            "x": 10,
                            "y": 2,
                            "type": "MOUNTAIN",
                        }
                    ],
                },
                weather=clear_weather(),
            )

    def test_scenario_rejects_duplicate_radar_ids(self):
        with self.assertRaisesRegex(
            ValueError,
            "radar_id",
        ):
            TacticalEnv.from_scenario(
                {
                    "radars": [
                        {
                            "radar_id": "radar_01",
                            "x": 2,
                            "y": 2,
                            "detection_range": 3.0,
                        },
                        {
                            "radar_id": "radar_01",
                            "x": 6,
                            "y": 6,
                            "detection_range": 3.0,
                        },
                    ],
                },
                weather=clear_weather(),
            )

    def test_more_than_three_radars_are_rejected(self):
        radars = [
            {
                "radar_id": f"radar_{index + 1:02d}",
                "x": index + 1,
                "y": 4,
                "detection_range": 3.0,
            }
            for index in range(4)
        ]

        with self.assertRaisesRegex(
            ValueError,
            "Maximum supported radar count is 3",
        ):
            TacticalEnv.from_scenario(
                {
                    "radars": radars,
                },
                weather=clear_weather(),
            )

    def test_step_returns_ctde_observations(self):
        with patch(
            "radar.radar.random.random",
            return_value=1.0,
        ):
            _, _, _, info = self.env.step(
                self.env.ACTION_STAY,
                hunter_action=self.env.ACTION_STAY,
            )

        self.assertIn("observations", info)
        self.assertIn("global_state", info)

        self.assertEqual(
            len(info["observations"]["scout"]),
            87,
        )
        self.assertEqual(
            len(info["observations"]["hunter"]),
            87,
        )
        self.assertEqual(
            len(info["global_state"]),
            101,
        )

        self.assertIn("reward_scout", info)
        self.assertIn("reward_hunter", info)
        self.assertIn("heading", info)

    def test_aircraft_heading_updates_with_movement(self):
        self.env.scout.move(7, 6)
        self.assertAlmostEqual(
            self.env.scout.state.heading,
            0.0,
        )

        self.env.scout.move(8, 6)
        self.assertAlmostEqual(
            self.env.scout.state.heading,
            90.0,
        )

        self.env.scout.move(8, 7)
        self.assertAlmostEqual(
            self.env.scout.state.heading,
            180.0,
        )

        self.env.scout.move(7, 7)
        self.assertAlmostEqual(
            self.env.scout.state.heading,
            270.0,
        )

    def test_hunter_accepts_explicit_movement_action(self):
        initial_fuel = self.env.hunter.state.fuel

        with patch(
            "radar.radar.random.random",
            return_value=1.0,
        ):
            _, _, _, info = self.env.step(
                self.env.ACTION_STAY,
                hunter_action=self.env.ACTION_UP,
            )

        self.assertEqual(
            (
                self.env.hunter.state.position.x,
                self.env.hunter.state.position.y,
            ),
            (8, 7),
        )
        self.assertEqual(
            self.env.hunter.state.fuel,
            initial_fuel - 1.0,
        )
        self.assertAlmostEqual(
            self.env.hunter.state.heading,
            0.0,
        )
        self.assertEqual(
            info["hunter_action"],
            self.env.ACTION_UP,
        )
    def test_hunter_detection_does_not_change_scout_reward(self):
        def run_step(radars):
            env = TacticalEnv(
                weather=clear_weather(),
                radars_config=radars,
            )
            env.reset()

            with patch(
                "radar.radar.random.random",
                return_value=0.0,
            ):
                _, _, _, info = env.step(
                    env.ACTION_STAY,
                    hunter_action=env.ACTION_STAY,
                )

            return info

        no_radar_info = run_step([])

        hunter_radar_info = run_step(
            [
                {
                    "radar_id": "hunter_radar",
                    "x": 8,
                    "y": 8,
                    "detection_range": 0.5,
                }
            ]
        )

        self.assertEqual(
            len(hunter_radar_info["radar_detections"]),
            0,
        )
        self.assertEqual(
            len(hunter_radar_info["hunter_radar_detections"]),
            1,
        )

        self.assertAlmostEqual(
            hunter_radar_info["reward_scout"],
            no_radar_info["reward_scout"],
        )

        self.assertLess(
            hunter_radar_info["reward_hunter"],
            no_radar_info["reward_hunter"],
        )

    def test_hunter_action_cost_is_in_shared_reward(self):
        def run_step(hunter_action):
            env = TacticalEnv(
                weather=clear_weather(),
                radars_config=[],
            )
            env.reset()

            _, reward, _, info = env.step(
                env.ACTION_STAY,
                hunter_action=hunter_action,
            )

            return reward, info

        stay_reward, stay_info = run_step(
            self.env.ACTION_STAY
        )
        jam_reward, jam_info = run_step(
            self.env.ACTION_JAM_SUPPRESS
        )

        self.assertLess(jam_reward, stay_reward)
        self.assertAlmostEqual(
            jam_reward - stay_reward,
            jam_info["reward_hunter"]
            - stay_info["reward_hunter"],
        )
    def test_scenario_rejects_position_outside_grid(self):
        with self.assertRaisesRegex(
            ValueError,
            "scout_start",
        ):
            TacticalEnv.from_scenario(
                {
                    "width": 10,
                    "height": 10,
                    "scout_start": [10, 0],
                },
                weather=clear_weather(),
            )

    def test_scenario_rejects_non_positive_cell_size(self):
        with self.assertRaisesRegex(
            ValueError,
            "cell_km",
        ):
            TacticalEnv.from_scenario(
                {
                    "cell_km": 0,
                },
                weather=clear_weather(),
            )

if __name__ == "__main__":
    unittest.main()
