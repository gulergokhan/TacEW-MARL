import torch.nn as nn


class DQNNetwork(nn.Module):

    def __init__(
        self,
        observation_size: int,
        action_size: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_size),
        )

    def forward(self, observation):
        return self.network(observation)