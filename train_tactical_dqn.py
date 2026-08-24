import numpy as np

from environment.tactical_env import TacticalEnv
from dqn.agent import DQNAgent


NUM_EPISODES = 500

OBSERVATION_SIZE = 57
ACTION_SIZE = 5


def main():

    env = TacticalEnv(
        width=10,
        height=10,
        latitude=39.9334,
        longitude=32.8597,
    )

    agent = DQNAgent(
        observation_size=OBSERVATION_SIZE,
        action_size=ACTION_SIZE,
    )

    episode_rewards = []
    episode_losses = []

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


if __name__ == "__main__":
    main()