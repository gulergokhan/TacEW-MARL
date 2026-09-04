import os
import numpy as np
import torch

from environment.tactical_env import TacticalEnv
from marl.algorithms.happo import HAPPO
from marl.buffer import MARLRolloutBuffer
from configs import marl_config as cfg


def get_ctde_observations(env):
    """
    Read decentralized observations and centralized state
    directly from TacticalEnv.
    """

    state = env._get_state()

    observations = env._get_agent_observations(state)

    return (
        observations["scout"],
        observations["hunter"],
        observations["global"],
    )


def main():
    print("=" * 60)
    print("TALON - HAPPO / CTDE TRAINING")
    print("=" * 60)

    print(f"Scout observation : {cfg.SCOUT_OBS_DIM}")
    print(f"Hunter observation: {cfg.HUNTER_OBS_DIM}")
    print(f"Global state     : {cfg.GLOBAL_STATE_DIM}")
    print(f"Action dimension : {cfg.ACTION_DIM}")
    print(f"Episodes         : {cfg.NUM_EPISODES}")
    print(f"Device           : {cfg.DEVICE}")
    print("=" * 60)

    env = TacticalEnv()
    happo = HAPPO()
    buffer = MARLRolloutBuffer()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    episode_rewards = []
    best_reward = -float("inf")

    total_steps = 0

    for episode in range(1, cfg.NUM_EPISODES + 1):

        env.reset()

        done = False
        episode_reward = 0.0
        episode_scout_reward = 0.0
        episode_hunter_reward = 0.0
        episode_steps = 0

        while not done:

            # --------------------------------------------------
            # Current CTDE observations
            # --------------------------------------------------

            scout_obs, hunter_obs, global_state = (
                get_ctde_observations(env)
            )

            # --------------------------------------------------
            # Central critic value
            # --------------------------------------------------

            global_tensor = torch.as_tensor(
                global_state,
                dtype=torch.float32,
                device=happo.device,
            ).unsqueeze(0)

            with torch.no_grad():
                value = happo.critic(
                    global_tensor
                ).squeeze().item()

            # --------------------------------------------------
            # Decentralized action selection
            # --------------------------------------------------

            (
                scout_action,
                hunter_action,
                scout_log_prob,
                hunter_log_prob,
            ) = happo.select_actions(
                scout_obs,
                hunter_obs,
            )

            # --------------------------------------------------
            # Environment step
            # --------------------------------------------------

            _, reward, done, info = env.step(
                scout_action=scout_action,
                hunter_action=hunter_action,
            )

            scout_reward = float(
                info.get("reward_scout", reward)
            )

            hunter_reward = float(
                info.get("reward_hunter", reward)
            )

            # --------------------------------------------------
            # Store transition
            # --------------------------------------------------

            buffer.add(
                scout_obs=scout_obs,
                hunter_obs=hunter_obs,
                global_state=global_state,
                scout_action=scout_action,
                hunter_action=hunter_action,
                scout_log_prob=scout_log_prob,
                hunter_log_prob=hunter_log_prob,
                reward=reward,
                scout_reward=scout_reward,
                hunter_reward=hunter_reward,
                value=value,
                done=done,
            )

            episode_reward += float(reward)
            episode_scout_reward += scout_reward
            episode_hunter_reward += hunter_reward

            episode_steps += 1
            total_steps += 1

            # --------------------------------------------------
            # Update after rollout
            # --------------------------------------------------

            if len(buffer) >= cfg.ROLLOUT_LENGTH:

                metrics = happo.update(
                    buffer.get()
                )

                print(
                    f"[UPDATE] "
                    f"step={total_steps} | "
                    f"buffer={len(buffer)} | "
                    f"critic={metrics['critic_loss']:.5f} | "
                    f"scout={metrics['scout_policy_loss']:.5f} | "
                    f"hunter={metrics['hunter_policy_loss']:.5f}"
                )

                buffer.clear()

        # ------------------------------------------------------
        # Episode statistics
        # ------------------------------------------------------

        episode_rewards.append(episode_reward)

        avg_reward = np.mean(
            episode_rewards[-20:]
        )

        print(
            f"Episode {episode:4d} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Avg20: {avg_reward:8.2f} | "
            f"Scout: {episode_scout_reward:8.2f} | "
            f"Hunter: {episode_hunter_reward:8.2f} | "
            f"Steps: {episode_steps:3d}"
        )

        # ------------------------------------------------------
        # Save best model
        # ------------------------------------------------------

        if episode_reward > best_reward:

            best_reward = episode_reward

            torch.save(
                {
                    "episode": episode,
                    "scout_actor": (
                        happo.scout_actor.state_dict()
                    ),
                    "hunter_actor": (
                        happo.hunter_actor.state_dict()
                    ),
                    "critic": (
                        happo.critic.state_dict()
                    ),
                    "best_reward": best_reward,
                },
                "models/happo_best.pth",
            )

            print(
                f"  -> Best HAPPO model saved "
                f"(reward={best_reward:.2f})"
            )

        # ------------------------------------------------------
        # Periodic checkpoint
        # ------------------------------------------------------

        if episode % 100 == 0:

            torch.save(
                {
                    "episode": episode,
                    "scout_actor": (
                        happo.scout_actor.state_dict()
                    ),
                    "hunter_actor": (
                        happo.hunter_actor.state_dict()
                    ),
                    "critic": (
                        happo.critic.state_dict()
                    ),
                    "episode_rewards": episode_rewards,
                },
                f"models/happo_episode_{episode}.pth",
            )

            print(
                f"  -> Checkpoint saved: "
                f"happo_episode_{episode}.pth"
            )

    # ----------------------------------------------------------
    # Final rollout update
    # ----------------------------------------------------------

    if len(buffer) > 0:

        metrics = happo.update(
            buffer.get()
        )

        print(
            f"[FINAL UPDATE] "
            f"critic={metrics['critic_loss']:.5f} | "
            f"scout={metrics['scout_policy_loss']:.5f} | "
            f"hunter={metrics['hunter_policy_loss']:.5f}"
        )

        buffer.clear()

    # ----------------------------------------------------------
    # Final model
    # ----------------------------------------------------------

    torch.save(
        {
            "episode": cfg.NUM_EPISODES,
            "scout_actor": (
                happo.scout_actor.state_dict()
            ),
            "hunter_actor": (
                happo.hunter_actor.state_dict()
            ),
            "critic": (
                happo.critic.state_dict()
            ),
            "episode_rewards": episode_rewards,
            "best_reward": best_reward,
        },
        "models/happo_final.pth",
    )

    print()
    print("=" * 60)
    print("HAPPO TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best reward : {best_reward:.2f}")
    print(
        f"Final avg20: "
        f"{np.mean(episode_rewards[-20:]):.2f}"
    )
    print("Saved:")
    print("  models/happo_best.pth")
    print("  models/happo_final.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()
