import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch

from agent.dqn_agent import DQNAgent
from environment.tactical_env import TacticalEnv


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "tactical_dqn.pth"
)

DEFAULT_LOG_PATH = (
    Path(__file__).resolve().parent
    / "dashboard_logs"
    / "evaluation_episodes.json"
)


def load_agent(env, model_path):
    observation = env.reset()
    agent = DQNAgent(
        state_size=len(observation),
        action_size=env.NUM_ACTIONS,
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run train_tactical_dqn.py first."
        )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )
    agent.model.load_state_dict(checkpoint)
    agent.target_model.load_state_dict(checkpoint)
    agent.model.eval()
    agent.epsilon = 0.0

    return agent


def evaluate(
    num_episodes=10,
    model_path=DEFAULT_MODEL_PATH,
    log_path=DEFAULT_LOG_PATH,
    verbose=True,
):
    model_path = Path(model_path)
    log_path = Path(log_path)

    env = TacticalEnv()
    agent = load_agent(env, model_path)

    rewards = []
    successes = 0
    detections = 0
    episode_logs = []

    for episode in range(1, num_episodes + 1):
        observation = env.reset()
        done = False
        total_reward = 0.0
        episode_detections = 0

        while not done:
            action = agent.select_action(observation, training=False)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            episode_detections += len(info.get("radar_detections", []))

        success = bool(info.get("goal_reached", False))
        rewards.append(total_reward)
        successes += int(success)
        detections += episode_detections
        episode_logs.append({
            "label": f"Evaluation episode {episode}",
            "steps": deepcopy(env.episode_log),
        })

        if verbose:
            print(
                f"Episode={episode:3d} | "
                f"Reward={total_reward:8.2f} | "
                f"Steps={len(env.episode_log):3d} | "
                f"Detections={episode_detections:3d} | "
                f"Success={success}"
            )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as file:
        json.dump({"episodes": episode_logs}, file, indent=2)

    summary = {
        "episodes": num_episodes,
        "average_reward": sum(rewards) / len(rewards),
        "success_rate": successes / num_episodes,
        "average_detections": detections / num_episodes,
        "dashboard_log": str(log_path),
    }

    print("\nTactical DQN Evaluation")
    print("------------------------")
    print(f"Episodes          : {summary['episodes']}")
    print(f"Average Reward    : {summary['average_reward']:.2f}")
    print(f"Success Rate      : {summary['success_rate']:.2%}")
    print(f"Avg. Detections   : {summary['average_detections']:.2f}")
    print(f"Dashboard Log     : {summary['dashboard_log']}")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a tactical DQN model.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        num_episodes=args.episodes,
        model_path=args.model,
        log_path=args.log,
        verbose=not args.quiet,
    )
