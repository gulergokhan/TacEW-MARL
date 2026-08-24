import unittest

from agent.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):

    def test_add_increases_buffer_size(self):
        buffer = ReplayBuffer(capacity=10)

        buffer.add(
            [0.0, 0.0],
            1,
            -1.0,
            [0.0, 0.1],
            False
        )

        self.assertEqual(len(buffer), 1)

    def test_capacity_removes_oldest_experience(self):
        buffer = ReplayBuffer(capacity=2)

        buffer.add([0], 0, 0, [1], False)
        buffer.add([1], 1, 0, [2], False)
        buffer.add([2], 2, 0, [3], True)

        stored_actions = [
            experience[1]
            for experience in buffer.buffer
        ]

        self.assertEqual(len(buffer), 2)
        self.assertEqual(stored_actions, [1, 2])

    def test_sample_returns_requested_batch_size(self):
        buffer = ReplayBuffer(capacity=10)

        buffer.add([0], 0, -1, [1], False)
        buffer.add([1], 1, 10, [2], True)

        states, actions, rewards, next_states, dones = (
            buffer.sample(2)
        )

        self.assertEqual(len(states), 2)
        self.assertEqual(len(actions), 2)
        self.assertEqual(len(rewards), 2)
        self.assertEqual(len(next_states), 2)
        self.assertEqual(len(dones), 2)

    def test_sample_fails_when_buffer_is_too_small(self):
        buffer = ReplayBuffer(capacity=10)
        buffer.add([0], 0, -1, [1], False)

        with self.assertRaises(ValueError):
            buffer.sample(2)


if __name__ == "__main__":
    unittest.main()
