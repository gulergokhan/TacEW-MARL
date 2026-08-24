from collections import deque
import random


class ReplayBuffer:

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                done,
            )
        )

    def sample(self, batch_size: int):

        batch = random.sample(
            self.buffer,
            batch_size,
        )

        states, actions, rewards, next_states, dones = zip(
            *batch
        )

        return (
            list(states),
            list(actions),
            list(rewards),
            list(next_states),
            list(dones),
        )

    def __len__(self):
        return len(self.buffer)