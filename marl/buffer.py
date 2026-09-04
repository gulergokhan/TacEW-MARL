import numpy as np


class MARLRolloutBuffer:
    """Rollout storage for multi-agent CTDE training."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.data = {
            "scout_obs": [],
            "hunter_obs": [],
            "global_state": [],
            "scout_actions": [],
            "hunter_actions": [],
            "scout_log_probs": [],
            "hunter_log_probs": [],
            "rewards": [],
            "scout_rewards": [],
            "hunter_rewards": [],
            "values": [],
            "dones": [],
        }

    def add(
        self,
        scout_obs,
        hunter_obs,
        global_state,
        scout_action,
        hunter_action,
        scout_log_prob,
        hunter_log_prob,
        reward,
        scout_reward,
        hunter_reward,
        value,
        done,
    ):
        self.data["scout_obs"].append(np.asarray(scout_obs, dtype=np.float32))
        self.data["hunter_obs"].append(np.asarray(hunter_obs, dtype=np.float32))
        self.data["global_state"].append(
            np.asarray(global_state, dtype=np.float32)
        )

        self.data["scout_actions"].append(int(scout_action))
        self.data["hunter_actions"].append(int(hunter_action))

        self.data["scout_log_probs"].append(float(scout_log_prob))
        self.data["hunter_log_probs"].append(float(hunter_log_prob))

        self.data["rewards"].append(float(reward))
        self.data["scout_rewards"].append(float(scout_reward))
        self.data["hunter_rewards"].append(float(hunter_reward))

        self.data["values"].append(float(value))
        self.data["dones"].append(bool(done))

    def __len__(self):
        return len(self.data["rewards"])

    def clear(self):
        self.data = {
            "scout_obs": [],
            "hunter_obs": [],
            "global_state": [],
            "scout_actions": [],
            "hunter_actions": [],
            "scout_log_probs": [],
            "hunter_log_probs": [],
            "rewards": [],
            "scout_rewards": [],
            "hunter_rewards": [],
            "values": [],
            "dones": [],
        }

    def get(self):
        return self.data
