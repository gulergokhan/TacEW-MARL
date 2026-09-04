import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    Policy network for a MARL agent.

    Input:
        87-dimensional decentralized observation

    Output:
        7 action logits
    """

    def __init__(
        self,
        observation_dim: int = 87,
        action_dim: int = 7,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)

    def get_action_distribution(
        self,
        observation: torch.Tensor,
    ) -> torch.distributions.Categorical:
        logits = self.forward(observation)
        return torch.distributions.Categorical(logits=logits)