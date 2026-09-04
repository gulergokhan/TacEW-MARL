import torch
import torch.nn as nn


class CentralCritic(nn.Module):
    """
    Centralized value network for CTDE.

    Input:
        101-dimensional global state

    Output:
        Scalar state value V(s)
    """

    def __init__(
        self,
        global_state_dim: int = 101,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.network(global_state)