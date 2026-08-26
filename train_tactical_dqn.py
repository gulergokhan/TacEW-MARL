import random
from copy import deepcopy
from collections import deque
from pathlib import Path

import numpy as np
import torch

from agent.dqn_agent import DQNAgent
from environment.tactical_env import TacticalEnv


DEFAULT_SEED = 42

EPISODES = 500
PRINT_INTERVAL = 10

ACTION_SIZE = 5

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "tactical_dqn.pth"
)


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        pass


def action_name(action):

    names = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT",
        4: "STAY",
    }

    return names.get(action, "UNKNOWN")


def print_radar_status(env):

    print("    RADARS:")

    for radar in env.radar_system.radars:

        distance = env.radar_system.distance(
            radar.position,
            env.scout.state.position,
        )

        print(
            f"      {radar.radar_id} | "
            f"Pos=({radar.position.x},{radar.position.y}) | "
            f"Distance={distance:.2f} | "
            f"Range={radar.detection_range:.1f} | "
            f"State={radar.state.value} | "
            f"Active={radar.active}"
        )


def evaluate_greedy(agent, env, verbose=False):

    state = env.reset()

    done = False

    total_reward = 0.0
    total_detections = 0
    steps = 0

    while not done:

        action = agent.select_action(
            state,
            training=False
        )

        (
            state,
            reward,
            done,
            info
        ) = env.step(action)

        total_reward += reward
        steps += 1

        detections = info.get(
            "radar_detections",
            []
        )

        total_detections += len(detections)

        if verbose:

            scout = env.scout.state.position

            print(
                f"    Step {steps:3d} | "
                f"Action={action_name(action):5s} | "
                f"Scout=({scout.x},{scout.y}) | "
                f"Reward={reward:6.2f}"
            )

            print_radar_status(env)

    return {
        "reward": total_reward,
        "detections": total_detections,
        "steps": steps,
        "goal_reached": info.get(
            "goal_reached",
            False
        ),
    }


def train(
    seed=DEFAULT_SEED,
    verbose=True
):

    set_seed(seed)

    if verbose:
        print(
            f"Training seed: {seed}"
        )

    env = TacticalEnv(
        width=10,
        height=10,
        latitude=39.9334,
        longitude=32.8597,
    )

    initial_observation = env.reset()

    observation_size = len(
        initial_observation
    )

    action_size = ACTION_SIZE

    if verbose:

        print(
            f"Observation Size: "
            f"{observation_size}"
        )

        print(
            f"Action Size: "
            f"{action_size}"
        )

    agent = DQNAgent(
        observation_size,
        action_size,

        learning_rate=0.001,
        gamma=0.99,

        buffer_size=10000,
        batch_size=64,

        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,

        target_update_freq=500,
    )

    reward_window = deque(
        maxlen=PRINT_INTERVAL
    )

    loss_window = deque(
        maxlen=PRINT_INTERVAL
    )

    detection_window = deque(
        maxlen=PRINT_INTERVAL
    )

    history = {
        "episode": [],
        "average_reward": [],
        "evaluation_reward": [],
        "detections": [],
        "epsilon": [],
        "loss": [],
        "goal_reached": [],
    }

    best_model_state = None
    best_score = float("-inf")
    best_episode = None

    for episode in range(
        1,
        EPISODES + 1
    ):

        state = env.reset()

        done = False

        total_reward = 0.0

        episode_losses = []

        episode_detections = 0

        while not done:

            action = agent.select_action(
                state,
                training=True
            )

            (
                next_state,
                reward,
                done,
                info
            ) = env.step(action)

            detections = info.get(
                "radar_detections",
                []
            )

            episode_detections += len(
                detections
            )

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            loss = agent.learn()

            if loss is not None:
                episode_losses.append(
                    loss
                )

            state = next_state

            total_reward += reward

        agent.decay_epsilon()

        if episode_losses:

            average_episode_loss = (
                sum(episode_losses)
                /
                len(episode_losses)
            )

        else:

            average_episode_loss = 0.0

        reward_window.append(
            total_reward
        )

        loss_window.append(
            average_episode_loss
        )

        detection_window.append(
            episode_detections
        )

        if episode % PRINT_INTERVAL == 0:

            average_reward = (
                sum(reward_window)
                /
                len(reward_window)
            )

            average_loss = (
                sum(loss_window)
                /
                len(loss_window)
            )

            average_detections = (
                sum(detection_window)
                /
                len(detection_window)
            )

            evaluation = evaluate_greedy(
                agent,
                env,
                verbose=False
            )

            evaluation_reward = evaluation[
                "reward"
            ]

            evaluation_detections = evaluation[
                "detections"
            ]

            goal_reached = evaluation[
                "goal_reached"
            ]

            history["episode"].append(
                episode
            )

            history["average_reward"].append(
                average_reward
            )

            history["evaluation_reward"].append(
                evaluation_reward
            )

            history["detections"].append(
                evaluation_detections
            )

            history["epsilon"].append(
                agent.epsilon
            )

            history["loss"].append(
                average_loss
            )

            history["goal_reached"].append(
                goal_reached
            )

            goal_bonus = 100.0 if goal_reached else 0.0

            score = (
                evaluation_reward
                + goal_bonus
                - (
                    evaluation_detections
                    * 2.0
                )
            )

            if score > best_score:

                best_score = score

                best_model_state = deepcopy(
                    agent.model.state_dict()
                )

                best_episode = episode

            if verbose:

                print(
                    f"Episode {episode:4d} | "
                    f"Avg Reward: "
                    f"{average_reward:8.2f} | "
                    f"Eval Reward: "
                    f"{evaluation_reward:8.2f} | "
                    f"Detections: "
                    f"{evaluation_detections:3d} | "
                    f"Goal: "
                    f"{str(goal_reached):5s} | "
                    f"Loss: "
                    f"{average_loss:.5f} | "
                    f"Epsilon: "
                    f"{agent.epsilon:.3f}"
                )

    if best_model_state is not None:

        agent.model.load_state_dict(
            best_model_state
        )

        agent.target_model.load_state_dict(
            best_model_state
        )

    history["best_episode"] = (
        best_episode
    )

    history["best_evaluation_reward"] = (
        best_score
    )

    return agent, history


def main():

    print(
        "Tactical DQN Training"
    )

    print(
        "======================"
    )

    agent, history = train(
        seed=DEFAULT_SEED,
        verbose=True
    )

    print()

    print(
        "Training Complete"
    )

    print(
        "======================"
    )

    print(
        "Best Episode:",
        history["best_episode"]
    )

    print(
        "Best Evaluation Score:",
        history["best_evaluation_reward"]
    )

    print(
        "Final Epsilon:",
        agent.epsilon
    )

    if history["evaluation_reward"]:

        print(
            "Final Evaluation Reward:",
            history[
                "evaluation_reward"
            ][-1]
        )

    print()

    print(
        "Final Greedy Evaluation"
    )

    print(
        "======================="
    )

    env = TacticalEnv(
        width=10,
        height=10,
        latitude=39.9334,
        longitude=32.8597,
    )

    evaluation = evaluate_greedy(
        agent,
        env,
        verbose=True
    )

    print()

    print(
        f"Final Reward: "
        f"{evaluation['reward']:.2f}"
    )

    print(
        f"Steps: "
        f"{evaluation['steps']}"
    )

    print(
        f"Detections: "
        f"{evaluation['detections']}"
    )

    print(
        f"Goal Reached: "
        f"{evaluation['goal_reached']}"
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        agent.model.state_dict(),
        MODEL_PATH
    )

    print()

    print(
        "Model saved to:",
        MODEL_PATH
    )


if __name__ == "__main__":

    main()