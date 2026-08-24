from pathlib import Path

import torch

from environment.tactical_env import TacticalEnv
from dqn.agent import DQNAgent


NUM_EPISODES = 10

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "tactical_dqn.pth"
)

OBSERVATION_SIZE = 78
ACTION_SIZE = 5


def evaluate():

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

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    agent.online_network.load_state_dict(
        checkpoint
    )

    agent.online_network.eval()

    # Evaluation sırasında exploration yok
    agent.epsilon = 0.0

    episode_rewards = []

    print()
    print("Tactical DQN Evaluation")
    print("========================")

    for episode in range(1, NUM_EPISODES + 1):

        observation = env.reset()

        done = False
        total_reward = 0.0
        step = 0

        print()
        print(f"Episode {episode}")
        print("----------------------")

        while not done:

            action = agent.select_action(
                observation,
                training=False,
            )

            (
                next_observation,
                reward,
                done,
                info,
            ) = env.step(action)

            observation = next_observation

            total_reward += reward
            step += 1

            scout = env.scout.state

            print(
                f"Step {step:3d} | "
                f"Action: {action} | "
                f"Reward: {reward:6.2f} | "
                f"Scout: "
                f"({scout.position.x}, "
                f"{scout.position.y}) | "
                f"Fuel: {scout.fuel:6.2f}"
            )

        episode_rewards.append(
            total_reward
        )

        print(
            f"Episode Reward: {total_reward:.2f}"
        )

    average_reward = sum(
        episode_rewards
    ) / len(episode_rewards)

    print()
    print("Evaluation Complete")
    print("========================")
    print(
        f"Average Reward: {average_reward:.2f}"
    )


if __name__ == "__main__":
    evaluate()
