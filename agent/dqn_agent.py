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
        batch_size=64
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.batch_size = batch_size

        self.model = DQN(
            state_size,
            action_size
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        self.memory = ReplayBuffer(
            buffer_size
        )

    def select_action(self, state):

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
            states,
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
            next_states,
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

        # Next Q-values
        with torch.no_grad():

            next_q_values = self.model(
                next_states
            ).max(1)[0]

        # Q-learning target
        target_q_values = rewards + (
            self.gamma *
            next_q_values *
            (1 - dones)
        )

        # Loss
        loss = nn.MSELoss()(
            current_q_values,
            target_q_values
        )

        # Backpropagation
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item() 