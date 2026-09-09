import unittest

from environment.gridworld import GridWorld
from configs.environment_config import (
    DISTANCE_REWARD_SCALE,
    GOAL_REWARD,
    STEP_REWARD,
)

class TestGridWorld(unittest.TestCase):

    def setUp(self):
        self.env = GridWorld()

    def test_reset_returns_initial_state(self):
        observation = self.env.reset()

        self.assertEqual(self.env.scout.position, (0, 0))
        self.assertTrue(self.env.scout.alive)
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(len(observation), 8)

    def test_invalid_move_keeps_same_position(self):
        self.env.reset()

        observation, reward, done, info = self.env.step(0)

        self.assertEqual(self.env.scout.position, (0, 0))
        self.assertEqual(reward, -5)
        self.assertFalse(done)
        self.assertEqual(info["step"], 1)

    def test_radar_detection_ends_episode(self):
        self.env.reset()

        actions = [1, 1, 3, 3]

        for action in actions:
            observation, reward, done, info = (
                self.env.step(action)
            )

            if done:
                break

        self.assertTrue(info["detected"])
        self.assertTrue(done)
        self.assertFalse(self.env.scout.alive)

    def test_reaching_goal_ends_episode(self):
        self.env.reset()
        self.env.scout.position = (9, 8)

        observation, reward, done, info = self.env.step(3)

        self.assertEqual(self.env.scout.position, (9, 9))
        self.assertTrue(info["reached_goal"])
        self.assertTrue(done)
        expected_reward = (
            STEP_REWARD
            + DISTANCE_REWARD_SCALE
            + GOAL_REWARD
        )

        self.assertAlmostEqual(
            reward,
            expected_reward,
        )


if __name__ == "__main__":
    unittest.main()
