import random

import numpy as np
import torch
import torch.nn as nn

from dqn.network import DQNNetwork
from dqn.replay_buffer import ReplayBuffer


class DQNAgent:

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 10000,
        batch_size: int = 64,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
    ):
        self.observation_size = observation_size
        self.action_size = action_size

        self.gamma = gamma
        self.batch_size = batch_size

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.target_update_freq = target_update_freq
        self.learn_step_count = 0

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # Online network
        self.online_network = DQNNetwork(
            observation_size,
            action_size,
        ).to(self.device)

        # Target network
        self.target_network = DQNNetwork(
            observation_size,
            action_size,
        ).to(self.device)

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

        self.target_network.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate,
        )

        # Replay buffer
        self.memory = ReplayBuffer(
            buffer_size
        )

    def select_action(
        self,
        observation,
        training: bool = True,
    ):
        """
        Epsilon-greedy action selection.
        """

        # Exploration
        if (
            training
            and random.random() < self.epsilon
        ):
            return random.randrange(
                self.action_size
            )

        # Exploitation
        observation_tensor = torch.tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.online_network(
                observation_tensor
            )

        return int(
            torch.argmax(
                q_values,
                dim=1,
            ).item()
        )

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        """
        Store experience in replay buffer.
        """

        self.memory.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

    def learn(self):
        """
        Perform one DQN learning step.
        """

        # Not enough experiences yet
        if len(self.memory) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.memory.sample(
            self.batch_size
        )

        # Convert numpy arrays efficiently
        states = torch.tensor(
            np.array(states),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        )

        # Current Q-values
        current_q_values = (
            self.online_network(states)
            .gather(1, actions)
            .squeeze(1)
        )

        # Target Q-values
        with torch.no_grad():

            next_q_values = (
                self.target_network(next_states)
                .max(dim=1)
                .values
            )

            target_q_values = (
                rewards
                + self.gamma
                * next_q_values
                * (1 - dones)
            )

        # Calculate loss
        loss = nn.SmoothL1Loss()(
            current_q_values,
            target_q_values,
        )

        # Backpropagation
        self.optimizer.zero_grad()

        loss.backward()

        # Prevent exploding gradients
        nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        self.learn_step_count += 1

        # Update target network
        if (
            self.learn_step_count
            % self.target_update_freq
            == 0
        ):
            self.target_network.load_state_dict(
                self.online_network.state_dict()
            )

        return loss.item()

    def decay_epsilon(self):
        """
        Reduce exploration over time.
        """

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )