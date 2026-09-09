import numpy as np
import torch
import torch.nn.functional as F

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

    Both actors and the centralized critic learn from the shared
    team reward. Agent-specific rewards are retained for diagnostics.
    """

    def __init__(self):
        self.device = torch.device(cfg.DEVICE)
        self.entropy_coef = cfg.ENTROPY_COEF

        # --------------------------------------------------
        # Actors
        # --------------------------------------------------

        self.scout_actor = Actor(
            observation_dim=cfg.SCOUT_OBS_DIM,
            action_dim=cfg.SCOUT_ACTION_DIM,
            hidden_dim=cfg.HIDDEN_DIM,
        ).to(self.device)

        self.hunter_actor = Actor(
            observation_dim=cfg.HUNTER_OBS_DIM,
            action_dim=cfg.HUNTER_ACTION_DIM,
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

    def set_entropy_coef(self, value):
        """Update the exploration bonus used by actor losses."""
        self.entropy_coef = float(value)

    # ==================================================
    # GAE
    # ==================================================

    @staticmethod
    def compute_gae(rewards, values, dones, next_value=0.0,):
        """
        Generalized Advantage Estimation.

        A_t = delta_t + gamma * lambda * A_{t+1}

        Returns:
            advantages
            returns
        """

        rewards = np.asarray(
            rewards,
            dtype=np.float32,
        )

        values = np.asarray(
            values,
            dtype=np.float32,
        )

        dones = np.asarray(
            dones,
            dtype=np.float32,
        )

        advantages = np.zeros_like(
            rewards,
            dtype=np.float32,
        )

        gae = 0.0


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
    # Advantage normalization
    # ==================================================

    @staticmethod
    def normalize_advantages(advantages):
        """
        Avantajları güvenli biçimde normalize eder.
        Tek elemanlı rollout durumunda NaN oluşmasını engeller.
        """

        advantages = torch.as_tensor(
            advantages,
            dtype=torch.float32,
        )

        if advantages.numel() <= 1:
            return torch.zeros_like(advantages)

        return (
            advantages - advantages.mean()
        ) / (
            advantages.std(unbiased=False) + 1e-8
        )

    # ==================================================
    # Tensor preparation
    # ==================================================

    def _prepare_batch(self, rollout, next_value=0.0,):

        scout_obs = torch.as_tensor(
            np.asarray(
                rollout["scout_obs"]
            ),
            dtype=torch.float32,
            device=self.device,
        )

        hunter_obs = torch.as_tensor(
            np.asarray(
                rollout["hunter_obs"]
            ),
            dtype=torch.float32,
            device=self.device,
        )

        global_state = torch.as_tensor(
            np.asarray(
                rollout["global_state"]
            ),
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

        # --------------------------------------------------
        # Shared reward
        # --------------------------------------------------

        # HAPPO tam iş birlikçi bir görevde ortak ödülü kullanır.
        shared_rewards = rollout["rewards"]
        values = rollout["values"]
        dones = rollout["dones"]

        shared_advantages, returns = self.compute_gae(
            shared_rewards,
            values,
            dones,
            next_value=next_value,
        )

        # İki aktör aynı takım avantajını kullanır.
        # Ajanların ayrı ödülleri yalnızca izleme ve raporlama için tutulur.
        scout_advantages = shared_advantages.copy()
        hunter_advantages = shared_advantages.copy()

        # --------------------------------------------------
        # Convert to tensors
        # --------------------------------------------------

        shared_advantages = torch.as_tensor(
            shared_advantages,
            dtype=torch.float32,
            device=self.device,
        )

        scout_advantages = torch.as_tensor(
            scout_advantages,
            dtype=torch.float32,
            device=self.device,
        )

        hunter_advantages = torch.as_tensor(
            hunter_advantages,
            dtype=torch.float32,
            device=self.device,
        )

        returns = torch.as_tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )

        # --------------------------------------------------
        # Normalize each agent independently
        # --------------------------------------------------

        scout_advantages = self.normalize_advantages(
            scout_advantages
        )

        hunter_advantages = self.normalize_advantages(
            hunter_advantages
        )

        shared_advantages = self.normalize_advantages(
            shared_advantages
        )

        return {
            "scout_obs": scout_obs,
            "hunter_obs": hunter_obs,
            "global_state": global_state,

            "scout_actions": scout_actions,
            "hunter_actions": hunter_actions,

            "old_scout_log_probs": old_scout_log_probs,
            "old_hunter_log_probs": old_hunter_log_probs,

            "scout_advantages": scout_advantages,
            "hunter_advantages": hunter_advantages,
            "shared_advantages": shared_advantages,

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

        If correction_ratio is supplied, it is applied
        to the current policy ratio for HAPPO-style
        sequential agent updates.
        """

        total_loss = 0.0
        total_entropy = 0.0

        batch_size = observations.shape[0]

        indices = np.arange(
            batch_size
        )

        for _ in range(cfg.PPO_EPOCHS):

            np.random.shuffle(
                indices
            )

            for start in range(
                0,
                batch_size,
                cfg.MINIBATCH_SIZE,
            ):

                batch_indices = indices[
                    start:start
                    + cfg.MINIBATCH_SIZE
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

                # --------------------------------------------------
                # Current policy
                # --------------------------------------------------

                distribution = (
                    actor.get_action_distribution(
                        obs_batch
                    )
                )

                new_log_probs = (
                    distribution.log_prob(
                        actions_batch
                    )
                )

                entropy = (
                    distribution.entropy()
                    .mean()
                )

                # --------------------------------------------------
                # PPO policy ratio
                # --------------------------------------------------

                policy_ratio = torch.exp(
                    new_log_probs - old_log_batch
                )

                # Önceden güncellenmiş ajanların etkisi clipping
                # işleminden ayrı tutulmalıdır.
                if correction_ratio is None:
                    weighted_advantage = adv_batch
                else:
                    weighted_advantage = (
                        correction_ratio[idx].detach()
                        * adv_batch
                    )

                clipped_policy_ratio = torch.clamp(
                    policy_ratio,
                    1.0 - cfg.CLIP_EPSILON,
                    1.0 + cfg.CLIP_EPSILON,
                )

                surrogate_1 = (
                    policy_ratio
                    * weighted_advantage
                )

                surrogate_2 = (
                    clipped_policy_ratio
                    * weighted_advantage
                )

                policy_loss = -torch.min(
                    surrogate_1,
                    surrogate_2,
                ).mean()

                # --------------------------------------------------
                # Entropy bonus
                # --------------------------------------------------

                loss = (
                    policy_loss
                    - self.entropy_coef
                    * entropy
                )

                optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(),
                    max_norm=0.5,
                )

                optimizer.step()

                total_loss += (
                    policy_loss.item()
                )

                total_entropy += (
                    entropy.item()
                )

        return {
            "policy_loss": total_loss,
            "entropy": total_entropy,
        }

    # ==================================================
    # HAPPO update
    # ==================================================

    def update(self, rollout, next_value=0.0,):
        """
        Perform one CTDE HAPPO update.

        Update order:

            1. Central critic
            2. Scout actor
            3. Hunter actor

        The critic and both actors use the shared team reward.
        Sequential actor updates apply the HAPPO correction ratio.
        """

        if len(rollout["rewards"]) == 0:

            return {
                "critic_loss": 0.0,
                "scout_policy_loss": 0.0,
                "hunter_policy_loss": 0.0,
            }

        batch = self._prepare_batch(
            rollout,
            next_value=next_value,
        )

        # ==================================================
        # 1. Central critic update
        # ==================================================

        critic_loss_total = 0.0

        for _ in range(
            cfg.PPO_EPOCHS
        ):

            predicted_values = (
                self.critic(
                    batch["global_state"]
                ).squeeze(-1)
            )

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

            critic_loss_total += (
                critic_loss.item()
            )

        # ==================================================
        # 2. Sequential actor updates
        # ==================================================

        actor_data = {
            "scout": {
                "actor": self.scout_actor,
                "optimizer": self.scout_optimizer,
                "observations": batch["scout_obs"],
                "actions": batch["scout_actions"],
                "old_log_probs": batch["old_scout_log_probs"],
                "advantages": batch["scout_advantages"],
            },
            "hunter": {
                "actor": self.hunter_actor,
                "optimizer": self.hunter_optimizer,
                "observations": batch["hunter_obs"],
                "actions": batch["hunter_actions"],
                "old_log_probs": batch["old_hunter_log_probs"],
                "advantages": batch["hunter_advantages"],
            },
        }

        update_order = ["scout", "hunter"]
        np.random.shuffle(update_order)

        correction_ratio = torch.ones_like(
            batch["shared_advantages"]
        )

        actor_results = {}

        for agent_name in update_order:
            data = actor_data[agent_name]

            actor_results[agent_name] = self._update_actor(
                actor=data["actor"],
                optimizer=data["optimizer"],
                observations=data["observations"],
                actions=data["actions"],
                old_log_probs=data["old_log_probs"],
                advantages=data["advantages"],
                correction_ratio=correction_ratio,
            )

            with torch.no_grad():
                distribution = (
                    data["actor"].get_action_distribution(
                        data["observations"]
                    )
                )

                new_log_probs = distribution.log_prob(
                    data["actions"]
                )

                agent_ratio = torch.exp(
                    new_log_probs
                    - data["old_log_probs"]
                )

                correction_ratio = (
                    correction_ratio
                    * agent_ratio
                )

        scout_result = actor_results["scout"]
        hunter_result = actor_results["hunter"]

        # ==================================================
        # Results
        # ==================================================

        return {
            "critic_loss": (
                critic_loss_total
                / cfg.PPO_EPOCHS
            ),

            "scout_policy_loss": (
                scout_result[
                    "policy_loss"
                ]
            ),

            "hunter_policy_loss": (
                hunter_result[
                    "policy_loss"
                ]
            ),

            "scout_entropy": (
                scout_result[
                    "entropy"
                ]
            ),

            "hunter_entropy": (
                hunter_result[
                    "entropy"
                ]
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
                self.scout_actor
                .get_action_distribution(
                    scout_obs
                )
            )

            hunter_distribution = (
                self.hunter_actor
                .get_action_distribution(
                    hunter_obs
                )
            )

            scout_action = (
                scout_distribution.sample()
            )

            hunter_action = (
                hunter_distribution.sample()
            )

            scout_log_prob = (
                scout_distribution.log_prob(
                    scout_action
                )
            )

            hunter_log_prob = (
                hunter_distribution.log_prob(
                    hunter_action
                )
            )

        return (
            int(
                scout_action.item()
            ),
            int(
                hunter_action.item()
            ),
            float(
                scout_log_prob.item()
            ),
            float(
                hunter_log_prob.item()
            ),
        )
