import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from configs import marl_config as cfg
from environment.tactical_env import TacticalEnv
from marl.algorithms.happo import HAPPO
from marl.buffer import MARLRolloutBuffer
from marl.execution import (
    apply_hunter_progress_guard,
    apply_scout_jamming_guard,
    select_guarded_dqn_actions,
)
from radar.models import RadarState
from train_happo import (
    build_expert_radar_layout,
    evaluate_existing_best,
    get_curriculum_radars,
    get_entropy_coefficient,
    get_hunter_expert_action,
    get_training_paths,
    get_training_scenario,
    validation_score,
)


class TestHAPPO(unittest.TestCase):

    def setUp(self):
        self.original_epochs = cfg.PPO_EPOCHS
        self.original_batch_size = cfg.MINIBATCH_SIZE

        cfg.PPO_EPOCHS = 1
        cfg.MINIBATCH_SIZE = 4

    def tearDown(self):
        cfg.PPO_EPOCHS = self.original_epochs
        cfg.MINIBATCH_SIZE = self.original_batch_size

    def test_observation_and_action_dimensions(self):
        env = TacticalEnv()
        env.reset()

        observations = env._get_agent_observations(
            env._get_state()
        )

        self.assertEqual(
            len(observations["scout"]),
            cfg.SCOUT_OBS_DIM,
        )
        self.assertEqual(
            len(observations["hunter"]),
            cfg.HUNTER_OBS_DIM,
        )
        self.assertEqual(
            len(observations["global"]),
            cfg.GLOBAL_STATE_DIM,
        )

        happo = HAPPO()

        scout_action, hunter_action, _, _ = (
            happo.select_actions(
                observations["scout"],
                observations["hunter"],
            )
        )

        self.assertIn(
            scout_action,
            range(cfg.SCOUT_ACTION_DIM),
        )
        self.assertIn(
            hunter_action,
            range(cfg.HUNTER_ACTION_DIM),
        )

    def test_hunter_progress_guard_replaces_regressive_action(self):
        env = TacticalEnv(
            scout_start_position=(4, 4),
            hunter_start_position=(5, 4),
            strike_point=(1, 4),
            terrain_cells=[],
            radars_config=[],
        )
        env.reset()

        guarded_action = apply_hunter_progress_guard(
            env,
            scout_action=env.ACTION_LEFT,
            hunter_action=env.ACTION_RIGHT,
        )

        self.assertEqual(
            guarded_action,
            env.ACTION_LEFT,
        )

    def test_hunter_progress_guard_preserves_escort_range(self):
        env = TacticalEnv(
            scout_start_position=(9, 9),
            hunter_start_position=(5, 4),
            strike_point=(1, 4),
            terrain_cells=[],
            radars_config=[],
        )
        env.reset()

        guarded_action = apply_hunter_progress_guard(
            env,
            scout_action=env.ACTION_STAY,
            hunter_action=env.ACTION_STAY,
        )

        self.assertEqual(
            guarded_action,
            env.ACTION_STAY,
        )

    def test_hunter_progress_guard_avoids_radar_center(self):
        env = TacticalEnv(
            scout_start_position=(6, 8),
            hunter_start_position=(3, 9),
            strike_point=(1, 8),
            terrain_cells=[],
            radars_config=[
                {
                    "radar_id": "radar_01",
                    "x": 3,
                    "y": 8,
                    "detection_range": 2.5,
                },
            ],
        )
        env.reset()

        guarded_action = apply_hunter_progress_guard(
            env,
            scout_action=env.ACTION_JAM_SUPPRESS,
            hunter_action=env.ACTION_UP,
        )

        self.assertEqual(
            guarded_action,
            env.ACTION_RIGHT,
        )

    def test_scout_jamming_guard_suppresses_imminent_threat(self):
        env = TacticalEnv(
            scout_start_position=(6, 7),
            hunter_start_position=(7, 8),
            strike_point=(1, 8),
            terrain_cells=[],
            radars_config=[
                {
                    "radar_id": "radar_01",
                    "x": 4,
                    "y": 8,
                    "detection_range": 3.0,
                },
            ],
        )
        env.reset()
        radar = env.radar_system.get_radar(
            "radar_01"
        )
        radar.get_track(
            env.hunter.state.aircraft_id
        ).state = RadarState.TRACK

        guarded_action = apply_scout_jamming_guard(
            env,
            scout_action=env.ACTION_LEFT,
        )

        self.assertEqual(
            guarded_action,
            env.ACTION_JAM_SUPPRESS,
        )

    def test_scout_jamming_guard_rejects_unneeded_jamming(self):
        env = TacticalEnv(
            scout_start_position=(2, 2),
            hunter_start_position=(3, 2),
            strike_point=(9, 9),
            terrain_cells=[],
            radars_config=[],
        )
        env.reset()

        guarded_action = apply_scout_jamming_guard(
            env,
            scout_action=env.ACTION_JAM_SUPPRESS,
        )

        self.assertIn(
            guarded_action,
            {
                env.ACTION_DOWN,
                env.ACTION_RIGHT,
            },
        )

    def test_scout_jamming_guard_replaces_regressive_move(self):
        env = TacticalEnv(
            scout_start_position=(4, 4),
            hunter_start_position=(5, 4),
            strike_point=(1, 4),
            terrain_cells=[],
            radars_config=[],
        )
        env.reset()

        guarded_action = apply_scout_jamming_guard(
            env,
            scout_action=env.ACTION_RIGHT,
        )

        self.assertEqual(
            guarded_action,
            env.ACTION_LEFT,
        )

    def test_scout_jamming_guard_allows_safe_radar_detour(self):
        env = TacticalEnv(
            scout_start_position=(5, 1),
            hunter_start_position=(5, 1),
            strike_point=(9, 9),
            terrain_cells=[],
            radars_config=[
                {
                    "radar_id": "radar_01",
                    "x": 6,
                    "y": 3,
                    "detection_range": 3.0,
                },
            ],
        )
        env.reset()

        guarded_action = apply_scout_jamming_guard(
            env,
            scout_action=env.ACTION_STAY,
        )

        self.assertEqual(
            guarded_action,
            env.ACTION_UP,
        )

    def test_guarded_dqn_actions_keep_custom_escort_together(self):
        env = TacticalEnv(
            scout_start_position=(0, 0),
            hunter_start_position=(1, 1),
            strike_point=(9, 9),
            terrain_cells=[],
            radars_config=[],
        )
        env.reset()

        scout_action, hunter_action = select_guarded_dqn_actions(
            env,
            scout_action=env.ACTION_STAY,
        )

        self.assertIn(
            scout_action,
            {env.ACTION_DOWN, env.ACTION_RIGHT},
        )
        self.assertIn(
            hunter_action,
            {env.ACTION_DOWN, env.ACTION_RIGHT},
        )

    def test_gae_uses_bootstrap_value(self):
        advantages, returns = HAPPO.compute_gae(
            rewards=[1.0],
            values=[0.5],
            dones=[False],
            next_value=0.25,
        )

        expected_advantage = (
            1.0
            + cfg.GAMMA * 0.25
            - 0.5
        )

        self.assertAlmostEqual(
            float(advantages[0]),
            expected_advantage,
            places=6,
        )
        self.assertAlmostEqual(
            float(returns[0]),
            expected_advantage + 0.5,
            places=6,
        )

    def test_curriculum_balances_targets_in_every_stage(self):
        stage_starts = [
            1,
            cfg.CURRICULUM_STAGE_1_END + 1,
            cfg.CURRICULUM_STAGE_2_END + 1,
        ]

        for first_episode in stage_starts:
            targets = [
                get_training_scenario(episode)[3]
                for episode in range(
                    first_episode,
                    first_episode
                    + len(cfg.TRAINING_STRIKE_POINTS) * 3,
                )
            ]

            for target in cfg.TRAINING_STRIKE_POINTS:
                self.assertEqual(
                    targets.count(target),
                    3,
                )

    def test_curriculum_cycles_training_radar_layouts(self):
        scenario_count = (
            len(cfg.TRAINING_STRIKE_POINTS)
            * len(cfg.TRAINING_START_FORMATIONS)
            * len(cfg.HAPPO_TRAINING_RADAR_LAYOUT_INDICES)
        )
        radar_layouts = [
            get_training_scenario(episode)[4]
            for episode in range(1, scenario_count + 1)
        ]

        for layout_index in (
            cfg.HAPPO_TRAINING_RADAR_LAYOUT_INDICES
        ):
            self.assertEqual(
                radar_layouts.count(layout_index),
                len(cfg.TRAINING_STRIKE_POINTS)
                * len(cfg.TRAINING_START_FORMATIONS),
            )

    def test_curriculum_uses_every_formation_from_stage_one(self):
        scenario_count = (
            len(cfg.TRAINING_STRIKE_POINTS)
            * len(cfg.TRAINING_START_FORMATIONS)
        )
        formations = [
            get_training_scenario(episode)[1:3]
            for episode in range(1, scenario_count + 1)
        ]

        for formation in cfg.TRAINING_START_FORMATIONS:
            self.assertEqual(
                formations.count(formation),
                len(cfg.TRAINING_STRIKE_POINTS),
            )

    def test_training_does_not_leak_holdout_formation(self):
        holdout_formation = ((9, 6), (9, 7))

        self.assertNotIn(
            holdout_formation,
            cfg.TRAINING_START_FORMATIONS,
        )

    def test_curriculum_increases_radar_complexity(self):
        forbidden = (
            (7, 7),
            (8, 8),
            (1, 8),
        )

        self.assertEqual(
            get_curriculum_radars(
                1,
                1,
                forbidden_positions=forbidden,
            ),
            [],
        )
        self.assertIsNone(
            get_curriculum_radars(
                2,
                1,
                forbidden_positions=forbidden,
            )
        )
        self.assertEqual(
            len(
                get_curriculum_radars(
                    3,
                    1,
                    forbidden_positions=forbidden,
                )
            ),
            3,
        )

    def test_entropy_coefficient_decays_across_training(self):
        start = get_entropy_coefficient(1)
        middle = get_entropy_coefficient(
            cfg.NUM_EPISODES // 2
        )
        end = get_entropy_coefficient(
            cfg.NUM_EPISODES
        )

        self.assertAlmostEqual(
            start,
            cfg.ENTROPY_COEF,
        )
        self.assertAlmostEqual(
            end,
            cfg.ENTROPY_COEF_END,
        )
        self.assertGreater(start, middle)
        self.assertGreater(middle, end)

    def test_scratch_training_uses_isolated_checkpoints(self):
        paths = get_training_paths("scratch")

        self.assertEqual(
            paths["best"],
            Path("models/happo_scratch_best.pth"),
        )
        self.assertEqual(
            paths["final"],
            Path("models/happo_scratch_final.pth"),
        )

    def test_resume_training_uses_production_checkpoints(self):
        paths = get_training_paths("resume")

        self.assertEqual(
            paths["best"],
            Path("models/happo_best.pth"),
        )
        self.assertEqual(
            paths["final"],
            Path("models/happo_final.pth"),
        )

    def test_hunter_expert_moves_toward_each_training_target(self):
        expected_actions = {
            (1, 8): TacticalEnv.ACTION_LEFT,
            (8, 0): TacticalEnv.ACTION_UP,
            (1, 1): TacticalEnv.ACTION_UP,
        }

        for strike_point, expected_action in expected_actions.items():
            env = TacticalEnv(
                hunter_start_position=(8, 8),
                strike_point=strike_point,
            )
            env.reset()

            self.assertEqual(
                get_hunter_expert_action(env),
                expected_action,
            )

    def test_synthetic_expert_radar_layout_is_valid_and_repeatable(self):
        forbidden = {
            (7, 7),
            (8, 8),
            (1, 8),
        }
        first = build_expert_radar_layout(
            1,
            forbidden_positions=forbidden,
        )
        second = build_expert_radar_layout(
            1,
            forbidden_positions=forbidden,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(
            len({radar["radar_id"] for radar in first}),
            3,
        )
        self.assertEqual(
            len({(radar["x"], radar["y"]) for radar in first}),
            3,
        )

        for radar in first:
            self.assertNotIn(
                (radar["x"], radar["y"]),
                forbidden,
            )

    def test_validation_score_prioritizes_worst_target(self):
        higher_mean_but_weak_target = {
            "worst_success_rate": 50.0,
            "mean_success_rate": 90.0,
            "average_reward": 80.0,
        }
        balanced_checkpoint = {
            "worst_success_rate": 75.0,
            "mean_success_rate": 80.0,
            "average_reward": 60.0,
        }

        self.assertGreater(
            validation_score(balanced_checkpoint),
            validation_score(
                higher_mean_but_weak_target
            ),
        )

    def test_validation_score_prioritizes_worst_radar_case(self):
        higher_mean_but_weak_radar = {
            "worst_success_rate": 100.0,
            "mean_success_rate": 100.0,
            "radar_success_rate": 99.0,
            "worst_radar_success_rate": 33.0,
            "average_reward": 130.0,
        }
        robust_checkpoint = {
            "worst_success_rate": 90.0,
            "mean_success_rate": 95.0,
            "radar_success_rate": 95.0,
            "worst_radar_success_rate": 80.0,
            "average_reward": 110.0,
        }

        self.assertGreater(
            validation_score(robust_checkpoint),
            validation_score(higher_mean_but_weak_radar),
        )

    def test_missing_existing_best_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_path = Path(directory) / "missing.pth"

            self.assertIsNone(
                evaluate_existing_best(missing_path)
            )

    def test_single_transition_update_is_finite(self):
        env = TacticalEnv()
        env.reset()

        happo = HAPPO()
        buffer = MARLRolloutBuffer()

        observations = env._get_agent_observations(
            env._get_state()
        )

        global_tensor = torch.as_tensor(
            observations["global"],
            dtype=torch.float32,
            device=happo.device,
        ).unsqueeze(0)

        with torch.no_grad():
            value = (
                happo.critic(global_tensor)
                .squeeze()
                .item()
            )

        (
            scout_action,
            hunter_action,
            scout_log_prob,
            hunter_log_prob,
        ) = happo.select_actions(
            observations["scout"],
            observations["hunter"],
        )

        _, reward, done, info = env.step(
            scout_action=scout_action,
            hunter_action=hunter_action,
        )

        buffer.add(
            scout_obs=observations["scout"],
            hunter_obs=observations["hunter"],
            global_state=observations["global"],
            scout_action=scout_action,
            hunter_action=hunter_action,
            scout_log_prob=scout_log_prob,
            hunter_log_prob=hunter_log_prob,
            reward=reward,
            scout_reward=info["reward_scout"],
            hunter_reward=info["reward_hunter"],
            value=value,
            done=done,
        )

        metrics = happo.update(
            buffer.get(),
            next_value=0.0,
        )

        for metric_value in metrics.values():
            self.assertTrue(
                math.isfinite(metric_value)
            )

        for parameter in happo.scout_actor.parameters():
            self.assertTrue(
                torch.isfinite(parameter).all()
            )

        for parameter in happo.hunter_actor.parameters():
            self.assertTrue(
                torch.isfinite(parameter).all()
            )

        for parameter in happo.critic.parameters():
            self.assertTrue(
                torch.isfinite(parameter).all()
            )


if __name__ == "__main__":
    unittest.main()
