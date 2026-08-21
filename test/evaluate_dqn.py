from pathlib import Path
import torch

from environment.gridworld import GridWorld
from agent.dqn_agent import DQNAgent


NUM_EPISODES = 100
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "dqn_model.pth"
)


def evaluate():

    env = GridWorld()

    initial_state = env.reset()

    state_size = len(initial_state)
    action_size = 4

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size
    )

    # Eğitilmiş modeli yükle
    agent.model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
)

    agent.model.eval()
    
    # Evaluation sırasında exploration yok
    agent.epsilon = 0.0

    total_rewards = []
    total_steps = 0
    successful_episodes = 0
    detected_episodes = 0

    for episode in range(NUM_EPISODES):

        state = env.reset()

        done = False
        total_reward = 0
        steps = 0

        while not done:

            action = agent.select_action(state)

            next_state, reward, done, info = env.step(
                action
            )

            state = next_state

            total_reward += reward
            steps += 1

        total_rewards.append(total_reward)
        total_steps += steps

        if info["reached_goal"]:
            successful_episodes += 1

        if info["detected"]:
            detected_episodes += 1

    average_reward = (
        sum(total_rewards) / NUM_EPISODES
    )

    average_steps = (
        total_steps / NUM_EPISODES
    )

    success_rate = (
        successful_episodes / NUM_EPISODES
    ) * 100

    detection_rate = (
        detected_episodes / NUM_EPISODES
    ) * 100

    print("-" * 70)
    print("DQN Evaluation Results")
    print("-" * 70)

    print(
        f"Episodes       : {NUM_EPISODES}"
    )

    print(
        f"Average Reward : {average_reward:.2f}"
    )

    print(
        f"Average Steps  : {average_steps:.2f}"
    )

    print(
        f"Success Rate   : {success_rate:.2f}%"
    )

    print(
        f"Detection Rate : {detection_rate:.2f}%"
    )

    print("-" * 70)


if __name__ == "__main__":
    evaluate()