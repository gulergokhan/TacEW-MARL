import numpy as np
import os
import torch

from environment.tactical_env import TacticalEnv
from dqn.agent import DQNAgent


NUM_EPISODES = 500
ACTION_SIZE = 5


def main():

    env = TacticalEnv(
        width=10,
        height=10,
        latitude=39.9334,
        longitude=32.8597,
    )

    # Get observation size directly from environment
    initial_observation = env.reset()

    observation_size = len(initial_observation)

    print(
        f"Observation Size: {observation_size}"
    )

    print(
        f"Action Size: {ACTION_SIZE}"
    )

    agent = DQNAgent(
        observation_size=observation_size,
        action_size=ACTION_SIZE,
    )

    episode_rewards = []
    episode_losses = []

    print()
    print("Tactical DQN Training")
    print("======================")

    for episode in range(1, NUM_EPISODES + 1):

        observation = env.reset()

        done = False
        total_reward = 0.0
        losses = []

        while not done:

            action = agent.select_action(
                observation,
                training=True,
            )

            (
                next_observation,
                reward,
                done,
                info,
            ) = env.step(action)

            agent.remember(
                observation,
                action,
                reward,
                next_observation,
                done,
            )

            loss = agent.learn()

            if loss is not None:
                losses.append(loss)

            observation = next_observation

            total_reward += reward

        agent.decay_epsilon()

        episode_rewards.append(total_reward)

        if losses:
            episode_losses.append(
                float(np.mean(losses))
            )
        else:
            episode_losses.append(0.0)

        if episode % 10 == 0:

            average_reward = np.mean(
                episode_rewards[-10:]
            )

            average_loss = np.mean(
                episode_losses[-10:]
            )

            print(
                f"Episode {episode:4d} | "
                f"Reward: {total_reward:7.2f} | "
                f"Avg Reward: {average_reward:7.2f} | "
                f"Loss: {average_loss:.5f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    print()
    print("Training Complete")
    print("======================")

    print(
        "Final Epsilon:",
        agent.epsilon,
    )

    print(
        "Final Average Reward:",
        np.mean(episode_rewards[-10:]),
    )

    # Save trained model
    os.makedirs(
        "models",
        exist_ok=True,
    )

    torch.save(
        agent.online_network.state_dict(),
        "models/tactical_dqn.pth",
    )

    print(
        "Model saved to models/tactical_dqn.pth"
    )


if __name__ == "__main__":
    main()
