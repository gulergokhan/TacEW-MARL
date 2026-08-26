import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.replay_buffer import ReplayBuffer


class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, state):
        return self.network(state)


class DQNAgent:

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        gamma=0.99,
        buffer_size=10000,
        batch_size=64,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        target_update_freq=100
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.batch_size = batch_size

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.learn_step_count = 0

        self.model = DQN(
            state_size,
            action_size
        )

        self.target_model = DQN(
            state_size,
            action_size
        )

        self.target_model.load_state_dict(
            self.model.state_dict()
        )

        self.target_model.eval()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        self.memory = ReplayBuffer(
            buffer_size
        )

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state)

        action = torch.argmax(
            q_values,
            dim=1
        ).item()

        return action

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.add(
            state,
            action,
            reward,
            next_state,
            done
        )

    def learn(self):

        if len(self.memory) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.memory.sample(self.batch_size)

        states = torch.tensor(
            np.asarray(states),
            dtype=torch.float32
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32
        )

        next_states = torch.tensor(
            np.asarray(next_states),
            dtype=torch.float32
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32
        )

        # Current Q-values
        current_q_values = self.model(states).gather(
            1,
            actions
        ).squeeze(1)
        # Double DQN:
        # Online model en iyi aksiyonu seçer.
        # Target model seçilen aksiyonun değerini hesaplar
        # Next Q-values
        with torch.no_grad():

            next_actions = self.model(
                next_states
            ).argmax(dim=1, keepdim=True)
            next_q_values = self.target_model(
                next_states
            ).gather(1, next_actions).squeeze(1)

        # Q-learning target
        target_q_values = rewards + (
            self.gamma *
            next_q_values *
            (1 - dones)
        )

        # Loss
        loss = nn.SmoothL1Loss()(
            current_q_values,
            target_q_values
        )

        # Backpropagation
        self.optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=10.0
        )

        self.optimizer.step()

        self.learn_step_count += 1

        if (
            self.learn_step_count
            % self.target_update_freq
            == 0
        ):
            self.target_model.load_state_dict(
                self.model.state_dict()
            )

        return loss.item()
