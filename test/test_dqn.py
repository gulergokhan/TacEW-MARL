import math
import random
import unittest

import torch

from agent.dqn_agent import DQNAgent


class TestDQNAgent(unittest.TestCase):

    def setUp(self):
        random.seed(42)
        torch.manual_seed(42)

        self.agent = DQNAgent(
            state_size=2,
            action_size=4,
            batch_size=4,
            target_update_freq=2
        )

    def test_select_action_returns_valid_action(self):
        action = self.agent.select_action(
            [0.0, 0.0],
            training=False
        )

        self.assertIsInstance(action, int)
        self.assertIn(action, range(4))

    def test_learn_returns_none_when_memory_is_small(self):
        self.agent.remember(
            [0.0, 0.0],
            1,
            -1.0,
            [0.1, 0.0],
            False
        )

        self.assertEqual(len(self.agent.memory), 1)
        self.assertIsNone(self.agent.learn())

    def test_learn_returns_finite_loss(self):
        experiences = [
            ([0.0, 0.0], 0, -1.0, [0.1, 0.0], False),
            ([0.1, 0.0], 1, -1.0, [0.2, 0.0], False),
            ([0.2, 0.0], 2, 10.0, [0.3, 0.0], True),
            ([0.3, 0.0], 3, -1.0, [0.4, 0.0], False),
        ]

        for experience in experiences:
            self.agent.remember(*experience)

        loss = self.agent.learn()

        self.assertIsInstance(loss, float)
        self.assertTrue(math.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
