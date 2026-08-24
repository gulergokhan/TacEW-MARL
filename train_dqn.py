from copy import deepcopy
from collections import deque
from pathlib import Path

import random
import torch

from agent.dqn_agent import DQNAgent
from environment.gridworld import GridWorld

DEFAULT_SEED = 42
EPISODES = 500
PRINT_INTERVAL = 10
MODEL_PATH = (
    Path(__file__).resolve().parent / "dqn_model.pth"
)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def evaluate_greedy(agent):
    env = GridWorld()
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(
            state,
            training=False
        )

        state, reward, done, info = env.step(action)

        total_reward += reward

    return {
        "reward": total_reward,
        "success": info["reached_goal"]
    }


def train(seed=DEFAULT_SEED, verbose=True):

    set_seed(seed)
    if verbose:
        print(f"Training seed: {seed}")

    env = GridWorld()

    initial_state = env.reset()

    agent = DQNAgent(
        state_size=len(initial_state),
        action_size=4,
        learning_rate=0.001,
        gamma=0.99,
        buffer_size=10000,
        batch_size=64,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        target_update_freq= 500
    )

    reward_window = deque(maxlen=PRINT_INTERVAL)
    success_window = deque(maxlen=PRINT_INTERVAL)
    loss_window = deque(maxlen=PRINT_INTERVAL)

    history = {
        "episode": [],
        "average_reward": [],
        "success_rate": [],
        "epsilon": [],
        "loss": [],
        "evaluation_reward": [],
        "evaluation_success": []
    }

    best_model_state = None
    best_score = (-1, float("-inf"))
    best_episode = None

    for episode in range(1, EPISODES + 1):

        state = env.reset()
        done = False

        total_reward = 0
        episode_losses = []

        while not done:

            action = agent.select_action(
                state,
                training=True
            )

            next_state, reward, done, info = env.step(action)

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            loss = agent.learn()

            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        reached_goal = info["reached_goal"]

        average_episode_loss = (
            sum(episode_losses) / len(episode_losses)
            if episode_losses
            else 0.0
        )

        reward_window.append(total_reward)
        success_window.append(int(reached_goal))
        loss_window.append(average_episode_loss)

        if episode % PRINT_INTERVAL == 0:

            average_reward = (
                sum(reward_window) / len(reward_window)
            )

            success_rate = (
                sum(success_window) / len(success_window)
            ) * 100

            average_loss = (
                sum(loss_window) / len(loss_window)
            )

            history["episode"].append(episode)
            history["average_reward"].append(average_reward)
            history["success_rate"].append(success_rate)
            history["epsilon"].append(agent.epsilon)
            history["loss"].append(average_loss)

            evaluation = evaluate_greedy(agent)

            history["evaluation_reward"].append(
                evaluation["reward"]
            )

            history["evaluation_success"].append(
                int(evaluation["success"])
            )

            candidate_score = (
                int(evaluation["success"]),
                evaluation["reward"]
            )

            if candidate_score > best_score:

                best_score = candidate_score

                best_model_state = deepcopy(
                    agent.model.state_dict()
                )

                best_episode = episode

            if verbose:
                print(
                    f"Episode={episode:4} | "
                    f"Average Reward={average_reward:8.2f} | "
                    f"Success Rate={success_rate:6.2f}% | "
                    f"Epsilon={agent.epsilon:.4f} | "
                    f"Loss={average_loss:.4f}"
                )
    if best_model_state is not None:

        agent.model.load_state_dict(
            best_model_state
        )

        agent.target_model.load_state_dict(
            best_model_state
        )

    history["best_episode"] = best_episode
    history["best_evaluation_reward"] = (
        best_score[1]
    )

    return agent, history


if __name__ == "__main__":

    trained_agent, training_history = train(
        seed=DEFAULT_SEED,
        verbose=True
    )

    torch.save(
        trained_agent.model.state_dict(),
        MODEL_PATH
    )

    print(f"Model kaydedildi: {MODEL_PATH}")
