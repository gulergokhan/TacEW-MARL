import numpy as np
import torch

from environment.tactical_env import TacticalEnv
from marl.algorithms.happo import HAPPO
from configs import marl_config as cfg


MODEL_PATH = "models/happo_best.pth"
NUM_EPISODES = 100


def get_ctde_observations(env):
    """
    Get decentralized observations for evaluation.
    """

    state = env._get_state()
    observations = env._get_agent_observations(state)

    return (
        observations["scout"],
        observations["hunter"],
    )


def load_best_model(happo):
    """
    Load the trained HAPPO checkpoint.
    """

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=happo.device,
    )

    happo.scout_actor.load_state_dict(
        checkpoint["scout_actor"]
    )

    happo.hunter_actor.load_state_dict(
        checkpoint["hunter_actor"]
    )

    if "critic" in checkpoint:
        happo.critic.load_state_dict(
            checkpoint["critic"]
        )

    # Evaluation mode
    happo.scout_actor.eval()
    happo.hunter_actor.eval()
    happo.critic.eval()

    return checkpoint


def select_deterministic_actions(
    happo,
    scout_obs,
    hunter_obs,
):
    """
    Deterministic decentralized execution.

    No sampling.
    No exploration.

    The action with the highest probability
    is selected for each agent.
    """

    scout_tensor = torch.as_tensor(
        scout_obs,
        dtype=torch.float32,
        device=happo.device,
    ).unsqueeze(0)

    hunter_tensor = torch.as_tensor(
        hunter_obs,
        dtype=torch.float32,
        device=happo.device,
    ).unsqueeze(0)

    with torch.no_grad():

        scout_distribution = (
            happo.scout_actor.get_action_distribution(
                scout_tensor
            )
        )

        hunter_distribution = (
            happo.hunter_actor.get_action_distribution(
                hunter_tensor
            )
        )

        scout_action = torch.argmax(
            scout_distribution.probs,
            dim=-1,
        ).item()

        hunter_action = torch.argmax(
            hunter_distribution.probs,
            dim=-1,
        ).item()

    return int(scout_action), int(hunter_action)


def get_position(agent):
    """
    Safely retrieve an aircraft position.
    """

    try:
        state = agent.state

        if hasattr(state, "position"):
            return state.position

        if hasattr(state, "x") and hasattr(state, "y"):
            return (state.x, state.y)

    except Exception:
        pass

    return "N/A"


def evaluate():

    print("=" * 60)
    print("TALON - HAPPO EVALUATION")
    print("=" * 60)

    print(f"Model             : {MODEL_PATH}")
    print(f"Episodes          : {NUM_EPISODES}")
    print("Exploration       : OFF")
    print(f"Device            : {cfg.DEVICE}")
    print("=" * 60)

    # --------------------------------------------------
    # Environment and model
    # --------------------------------------------------

    env = TacticalEnv()
    happo = HAPPO()

    checkpoint = load_best_model(happo)

    print(
        f"Training episode  : "
        f"{checkpoint.get('episode', 'N/A')}"
    )

    print(
        f"Training reward   : "
        f"{checkpoint.get('best_reward', 'N/A')}"
    )

    print("-" * 60)

    # --------------------------------------------------
    # Evaluation storage
    # --------------------------------------------------

    rewards = []
    scout_rewards = []
    hunter_rewards = []
    episode_steps = []

    successful_episodes = 0
    mission_successes = 0
    radar_failures = 0

    # --------------------------------------------------
    # Evaluation episodes
    # --------------------------------------------------

    for episode in range(1, NUM_EPISODES + 1):

        env.reset()

        done = False

        total_reward = 0.0
        total_scout_reward = 0.0
        total_hunter_reward = 0.0

        steps = 0

        episode_radar_failure = False
        episode_mission_success = False
        episode_mission_failed = False

        # --------------------------------------------------
        # Diagnostic header for first episode
        # --------------------------------------------------

        if episode == 1:
            print()
            print("=" * 60)
            print("FIRST EPISODE DIAGNOSTIC")
            print("=" * 60)

        while not done:

            scout_obs, hunter_obs = (
                get_ctde_observations(env)
            )

            # ------------------------------------------
            # Deterministic action selection
            # ------------------------------------------

            scout_action, hunter_action = (
                select_deterministic_actions(
                    happo,
                    scout_obs,
                    hunter_obs,
                )
            )

            # ------------------------------------------
            # Environment step
            # ------------------------------------------

            _, reward, done, info = env.step(
                scout_action=scout_action,
                hunter_action=hunter_action,
            )

            scout_reward = float(
                info.get(
                    "reward_scout",
                    reward,
                )
            )

            hunter_reward = float(
                info.get(
                    "reward_hunter",
                    reward,
                )
            )

            # ------------------------------------------
            # Metrics
            # ------------------------------------------

            total_reward += float(reward)
            total_scout_reward += scout_reward
            total_hunter_reward += hunter_reward

            steps += 1

            # ------------------------------------------
            # Radar failure
            # ------------------------------------------

            if (
                info.get("lethal_hit", False)
                or
                info.get("hunter_lethal_hit", False)
            ):
                episode_radar_failure = True

            # ------------------------------------------
            # Mission status
            # ------------------------------------------

            if info.get("mission_success", False):
                episode_mission_success = True

            if info.get("mission_failed", False):
                episode_mission_failed = True

            # ------------------------------------------
            # First episode diagnostic
            # ------------------------------------------

            if episode == 1 and steps <= 20:

                radar_status = info.get(
                    "radar_status",
                    "N/A",
                )

                termination_reason = info.get(
                    "termination_reason",
                    None,
                )

                print(
                    f"Step {steps:3d} | "
                    f"Scout Action: {scout_action} | "
                    f"Hunter Action: {hunter_action} | "
                    f"Reward: {float(reward):6.2f} | "
                    f"Scout R: {scout_reward:6.2f} | "
                    f"Hunter R: {hunter_reward:6.2f}"
                )

                if steps <= 3 or done:
                    print(
                        f"          Radar: "
                        f"{radar_status} | "
                        f"Mission Success: "
                        f"{info.get('mission_success', False)} | "
                        f"Mission Failed: "
                        f"{info.get('mission_failed', False)} | "
                        f"Termination: "
                        f"{termination_reason}"
                    )

        # --------------------------------------------------
        # Episode complete
        # --------------------------------------------------

        rewards.append(total_reward)

        scout_rewards.append(
            total_scout_reward
        )

        hunter_rewards.append(
            total_hunter_reward
        )

        episode_steps.append(steps)

        if episode_radar_failure:
            radar_failures += 1

        if episode_mission_success:
            mission_successes += 1
            successful_episodes += 1

        # --------------------------------------------------
        # Episode output
        # --------------------------------------------------

        print(
            f"Episode {episode:3d} | "
            f"Reward: {total_reward:8.2f} | "
            f"Scout: {total_scout_reward:8.2f} | "
            f"Hunter: {total_hunter_reward:8.2f} | "
            f"Steps: {steps:3d} | "
            f"Success: "
            f"{'YES' if episode_mission_success else 'NO'}"
        )

    # --------------------------------------------------
    # Final statistics
    # --------------------------------------------------

    average_reward = float(
        np.mean(rewards)
    )

    success_rate = (
        successful_episodes
        / NUM_EPISODES
    ) * 100.0

    average_steps = float(
        np.mean(episode_steps)
    )

    average_scout_reward = float(
        np.mean(scout_rewards)
    )

    average_hunter_reward = float(
        np.mean(hunter_rewards)
    )

    radar_failure_rate = (
        radar_failures
        / NUM_EPISODES
    ) * 100.0

    mission_success_rate = (
        mission_successes
        / NUM_EPISODES
    ) * 100.0

    # --------------------------------------------------
    # Final results
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("HAPPO EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"Average Reward     : "
        f"{average_reward:.2f}"
    )

    print(
        f"Success Rate       : "
        f"{success_rate:.2f}%"
    )

    print(
        f"Average Steps      : "
        f"{average_steps:.2f}"
    )

    print(
        f"Scout Reward       : "
        f"{average_scout_reward:.2f}"
    )

    print(
        f"Hunter Reward      : "
        f"{average_hunter_reward:.2f}"
    )

    print(
        f"Radar Failure      : "
        f"{radar_failure_rate:.2f}%"
    )

    print(
        f"Mission Success    : "
        f"{mission_success_rate:.2f}%"
    )

    print("=" * 60)

    # --------------------------------------------------
    # Additional diagnostics
    # --------------------------------------------------

    print()
    print("ADDITIONAL DIAGNOSTICS")
    print("-" * 60)

    print(
        f"Successful episodes : "
        f"{successful_episodes}/{NUM_EPISODES}"
    )

    print(
        f"Radar failures      : "
        f"{radar_failures}/{NUM_EPISODES}"
    )

    print(
        f"Mission successes   : "
        f"{mission_successes}/{NUM_EPISODES}"
    )

    print(
        f"Min reward          : "
        f"{np.min(rewards):.2f}"
    )

    print(
        f"Max reward          : "
        f"{np.max(rewards):.2f}"
    )

    print(
        f"Reward std          : "
        f"{np.std(rewards):.2f}"
    )

    print("=" * 60)

    return {
        "average_reward": average_reward,
        "success_rate": success_rate,
        "average_steps": average_steps,
        "scout_reward": average_scout_reward,
        "hunter_reward": average_hunter_reward,
        "radar_failure_rate": radar_failure_rate,
        "mission_success_rate": mission_success_rate,
    }


if __name__ == "__main__":
    evaluate()