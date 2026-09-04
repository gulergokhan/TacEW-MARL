import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from marl.networks import Actor, CentralCritic
from configs import marl_config as cfg


class HAPPO:
    """
    Heterogeneous-Agent PPO (HAPPO) style trainer
    with Centralized Training and Decentralized Execution (CTDE).

    Agents:
        - Scout
        - Hunter

    Actors receive decentralized observations.
    Critic receives the centralized global state.

    Agents are updated sequentially rather than as
    completely independent PPO policies.
    """

    def __init__(self):
        self.device = torch.device(cfg.DEVICE)

        # --------------------------------------------------
        # Actors
        # --------------------------------------------------
        self.scout_actor = Actor(
            observation_dim=cfg.SCOUT_OBS_DIM,
            action_dim=cfg.ACTION_DIM,
            hidden_dim=cfg.HIDDEN_DIM,
        ).to(self.device)

        self.hunter_actor = Actor(
            observation_dim=cfg.HUNTER_OBS_DIM,
            action_dim=cfg.ACTION_DIM,
            hidden_dim=cfg.HIDDEN_DIM,
        ).to(self.device)

        # --------------------------------------------------
        # Centralized critic
        # --------------------------------------------------
        self.critic = CentralCritic(
            global_state_dim=cfg.GLOBAL_STATE_DIM,
            hidden_dim=cfg.HIDDEN_DIM,
        ).to(self.device)

        # --------------------------------------------------
        # Optimizers
        # --------------------------------------------------
        self.scout_optimizer = torch.optim.Adam(
            self.scout_actor.parameters(),
            lr=cfg.LEARNING_RATE_ACTOR,
        )

        self.hunter_optimizer = torch.optim.Adam(
            self.hunter_actor.parameters(),
            lr=cfg.LEARNING_RATE_ACTOR,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=cfg.LEARNING_RATE_CRITIC,
        )

    # ==================================================
    # GAE
    # ==================================================

    @staticmethod
    def compute_gae(rewards, values, dones):
        """
        Generalized Advantage Estimation.

        A_t = delta_t + gamma * lambda * A_{t+1}

        Returns:
            advantages
            returns
        """

        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)

        gae = 0.0
        next_value = 0.0

        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]

            delta = (
                rewards[step]
                + cfg.GAMMA * next_value * mask
                - values[step]
            )

            gae = (
                delta
                + cfg.GAMMA
                * cfg.GAE_LAMBDA
                * mask
                * gae
            )

            advantages[step] = gae
            next_value = values[step]

        returns = advantages + values

        return advantages, returns

    # ==================================================
    # Tensor preparation
    # ==================================================

    def _prepare_batch(self, rollout):
        scout_obs = torch.as_tensor(
            np.asarray(rollout["scout_obs"]),
            dtype=torch.float32,
            device=self.device,
        )

        hunter_obs = torch.as_tensor(
            np.asarray(rollout["hunter_obs"]),
            dtype=torch.float32,
            device=self.device,
        )

        global_state = torch.as_tensor(
            np.asarray(rollout["global_state"]),
            dtype=torch.float32,
            device=self.device,
        )

        scout_actions = torch.as_tensor(
            rollout["scout_actions"],
            dtype=torch.long,
            device=self.device,
        )

        hunter_actions = torch.as_tensor(
            rollout["hunter_actions"],
            dtype=torch.long,
            device=self.device,
        )

        old_scout_log_probs = torch.as_tensor(
            rollout["scout_log_probs"],
            dtype=torch.float32,
            device=self.device,
        )

        old_hunter_log_probs = torch.as_tensor(
            rollout["hunter_log_probs"],
            dtype=torch.float32,
            device=self.device,
        )

        rewards = rollout["rewards"]
        values = rollout["values"]
        dones = rollout["dones"]

        advantages, returns = self.compute_gae(
            rewards,
            values,
            dones,
        )

        advantages = torch.as_tensor(
            advantages,
            dtype=torch.float32,
            device=self.device,
        )

        returns = torch.as_tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )

        # Advantage normalization is standard PPO practice.
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std() + 1e-8)

        return {
            "scout_obs": scout_obs,
            "hunter_obs": hunter_obs,
            "global_state": global_state,
            "scout_actions": scout_actions,
            "hunter_actions": hunter_actions,
            "old_scout_log_probs": old_scout_log_probs,
            "old_hunter_log_probs": old_hunter_log_probs,
            "advantages": advantages,
            "returns": returns,
        }

    # ==================================================
    # Actor update
    # ==================================================

    def _update_actor(
        self,
        actor,
        optimizer,
        observations,
        actions,
        old_log_probs,
        advantages,
        correction_ratio=None,
    ):
        """
        PPO clipped actor update.

        For the second HAPPO agent, a correction ratio
        from the already-updated agent can be supplied.
        """

        total_loss = 0.0
        total_entropy = 0.0

        batch_size = observations.shape[0]

        indices = np.arange(batch_size)

        for _ in range(cfg.PPO_EPOCHS):
            np.random.shuffle(indices)

            for start in range(
                0,
                batch_size,
                cfg.MINIBATCH_SIZE,
            ):
                batch_indices = indices[
                    start:start + cfg.MINIBATCH_SIZE
                ]

                idx = torch.as_tensor(
                    batch_indices,
                    dtype=torch.long,
                    device=self.device,
                )

                obs_batch = observations[idx]
                actions_batch = actions[idx]
                old_log_batch = old_log_probs[idx]
                adv_batch = advantages[idx]

                distribution = actor.get_action_distribution(
                    obs_batch
                )

                new_log_probs = distribution.log_prob(
                    actions_batch
                )

                entropy = distribution.entropy().mean()

                ratio = torch.exp(
                    new_log_probs - old_log_batch
                )

                # HAPPO correction for previously updated agents.
                if correction_ratio is not None:
                    ratio = ratio * correction_ratio[idx]

                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - cfg.CLIP_EPSILON,
                    1.0 + cfg.CLIP_EPSILON,
                )

                surrogate_1 = ratio * adv_batch
                surrogate_2 = clipped_ratio * adv_batch

                policy_loss = -torch.min(
                    surrogate_1,
                    surrogate_2,
                ).mean()

                loss = (
                    policy_loss
                    - cfg.ENTROPY_COEF * entropy
                )

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(),
                    max_norm=0.5,
                )

                optimizer.step()

                total_loss += policy_loss.item()
                total_entropy += entropy.item()

        return {
            "policy_loss": total_loss,
            "entropy": total_entropy,
        }

    # ==================================================
    # HAPPO update
    # ==================================================

    def update(self, rollout):
        """
        Perform one CTDE HAPPO update.

        Update order:

            1. Central critic
            2. Scout actor
            3. Hunter actor

        The Hunter update uses a correction ratio based
        on the Scout's updated policy.
        """

        if len(rollout["rewards"]) == 0:
            return {
                "critic_loss": 0.0,
                "scout_policy_loss": 0.0,
                "hunter_policy_loss": 0.0,
            }

        batch = self._prepare_batch(rollout)

        # --------------------------------------------------
        # 1. Central critic update
        # --------------------------------------------------

        critic_loss_total = 0.0

        for _ in range(cfg.PPO_EPOCHS):
            predicted_values = self.critic(
                batch["global_state"]
            ).squeeze(-1)

            critic_loss = F.mse_loss(
                predicted_values,
                batch["returns"],
            )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                max_norm=0.5,
            )

            self.critic_optimizer.step()

            critic_loss_total += critic_loss.item()

        # --------------------------------------------------
        # 2. Scout update
        # --------------------------------------------------

        scout_result = self._update_actor(
            actor=self.scout_actor,
            optimizer=self.scout_optimizer,
            observations=batch["scout_obs"],
            actions=batch["scout_actions"],
            old_log_probs=batch["old_scout_log_probs"],
            advantages=batch["advantages"],
        )

        # --------------------------------------------------
        # 3. Calculate Scout correction ratio
        # --------------------------------------------------

        with torch.no_grad():
            scout_distribution = (
                self.scout_actor.get_action_distribution(
                    batch["scout_obs"]
                )
            )

            new_scout_log_probs = (
                scout_distribution.log_prob(
                    batch["scout_actions"]
                )
            )

            scout_ratio = torch.exp(
                new_scout_log_probs
                - batch["old_scout_log_probs"]
            )

        # --------------------------------------------------
        # 4. Hunter update
        # --------------------------------------------------

        hunter_result = self._update_actor(
            actor=self.hunter_actor,
            optimizer=self.hunter_optimizer,
            observations=batch["hunter_obs"],
            actions=batch["hunter_actions"],
            old_log_probs=batch["old_hunter_log_probs"],
            advantages=batch["advantages"],
            correction_ratio=scout_ratio,
        )

        return {
            "critic_loss": (
                critic_loss_total / cfg.PPO_EPOCHS
            ),
            "scout_policy_loss": (
                scout_result["policy_loss"]
            ),
            "hunter_policy_loss": (
                hunter_result["policy_loss"]
            ),
            "scout_entropy": (
                scout_result["entropy"]
            ),
            "hunter_entropy": (
                hunter_result["entropy"]
            ),
        }

    # ==================================================
    # Action selection
    # ==================================================

    def select_actions(
        self,
        scout_observation,
        hunter_observation,
    ):
        """
        Decentralized execution.

        Each actor only sees its own observation.
        """

        scout_obs = torch.as_tensor(
            scout_observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        hunter_obs = torch.as_tensor(
            hunter_observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            scout_distribution = (
                self.scout_actor.get_action_distribution(
                    scout_obs
                )
            )

            hunter_distribution = (
                self.hunter_actor.get_action_distribution(
                    hunter_obs
                )
            )

            scout_action = scout_distribution.sample()
            hunter_action = hunter_distribution.sample()

            scout_log_prob = scout_distribution.log_prob(
                scout_action
            )

            hunter_log_prob = hunter_distribution.log_prob(
                hunter_action
            )

        return (
            int(scout_action.item()),
            int(hunter_action.item()),
            float(scout_log_prob.item()),
            float(hunter_log_prob.item()),
        )
