import torch

from environment.tactical_env import TacticalEnv
from dqn.agent import DQNAgent


OBSERVATION_SIZE = 57
ACTION_SIZE = 5
MODEL_PATH = "models/tactical_dqn.pth"


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

    # Load trained model
    agent.online_network.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=agent.device,
        )
    )

    agent.online_network.eval()

    # No exploration during evaluation
    agent.epsilon = 0.0

    observation = env.reset()

    total_reward = 0.0
    step_count = 0
    done = False

    print("DQN Evaluation")
    print("======================")

    while not done and step_count < 100:

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

        x = env.scout.state.position.x
        y = env.scout.state.position.y
        fuel = env.scout.state.fuel

        print(
            f"Step {step_count + 1:3d} | "
            f"Action: {action} | "
            f"Reward: {reward:6.2f} | "
            f"Scout: ({x}, {y}) | "
            f"Fuel: {fuel:6.2f}"
        )

        observation = next_observation
        total_reward += reward
        step_count += 1

    print()
    print("Evaluation Complete")
    print("======================")
    print(f"Steps: {step_count}")
    print(f"Total Reward: {total_reward:.2f}")
    print(
        f"Final Position: "
        f"({env.scout.state.position.x}, "
        f"{env.scout.state.position.y})"
    )
    print(f"Remaining Fuel: {env.scout.state.fuel:.2f}") psr ssr 


if __name__ == "__main__":
    main()