from configs.environment_config import (
    GRID_SIZE,
    SCOUT_START_POSITION,
    GOAL_POSITION,
    MAX_STEPS,
    RADAR_DETECTION_RANGE,
    STEP_REWARD,
    DETECTION_PENALTY,
    GOAL_REWARD,
    DESTRUCTION_PENALTY,
    INVALID_MOVE_PENALTY,
)

from environment.entities.scout import Scout
from environment.entities.radar import Radar


class GridWorld:

    def __init__(self):
        self.grid_size = GRID_SIZE
        self.max_steps = MAX_STEPS

        self.scout = Scout(SCOUT_START_POSITION)

        self.radars = [
            Radar((3, 3), RADAR_DETECTION_RANGE),
            Radar((6, 6), RADAR_DETECTION_RANGE),
        ]

        self.goal = GOAL_POSITION

        self.current_step = 0

    def reset(self):
        """
        Environment'ı başlangıç durumuna getirir.
        """

        self.scout = Scout(SCOUT_START_POSITION)

        for radar in self.radars:
            radar.state = "SAFE"

        self.current_step = 0

        return self.get_observation()

    def step(self, action):
        """
        Agent bir action gerçekleştirir.

        Returns:
            observation
            reward
            done
            info
        """

        self.current_step += 1

        old_position = self.scout.position

        new_position = self._calculate_new_position(
            old_position,
            action
        )

        # Grid dışına çıkmaya çalışıyorsa
        if not self._is_valid_position(new_position):

            reward = INVALID_MOVE_PENALTY

            new_position = old_position

        else:
            self.scout.move(new_position)

            reward = STEP_REWARD

        # Radar detection kontrolü
        detected = self._check_radar_detection()

        if detected:
            reward += DETECTION_PENALTY

        # Goal kontrolü
        reached_goal = self.scout.position == self.goal

        if reached_goal:
            reward += GOAL_REWARD

        # Şimdilik detection = destruction kabul ediyoruz.
        if detected:
            self.scout.destroy()
            reward += DESTRUCTION_PENALTY

        # Episode bitiş kontrolü
        done = (
            reached_goal
            or not self.scout.alive
            or self.current_step >= self.max_steps
        )

        observation = self.get_observation()

        info = {
            "step": self.current_step,
            "scout_position": self.scout.position,
            "detected": detected,
            "reached_goal": reached_goal,
        }

        return observation, reward, done, info

    def get_observation(self):
        """
        Environment state'ini RL agent'ın kullanabileceği
        numerical vector haline getirir.
        """

        observation = [
            self.scout.position[0] / (self.grid_size - 1),
            self.scout.position[1] / (self.grid_size - 1),
        ]

        for radar in self.radars:

            observation.append(
                radar.position[0] / (self.grid_size - 1)
            )

            observation.append(
                radar.position[1] / (self.grid_size - 1)
            )

        observation.extend([
            self.goal[0] / (self.grid_size - 1),
            self.goal[1] / (self.grid_size - 1),
        ])

        return observation

    def _calculate_new_position(self, position, action):

        row, col = position

        if action == 0:       # UP
            row -= 1

        elif action == 1:     # DOWN
            row += 1

        elif action == 2:     # LEFT
            col -= 1

        elif action == 3:     # RIGHT
            col += 1

        else:
            raise ValueError(f"Invalid action: {action}")

        return row, col

    def _is_valid_position(self, position):

        row, col = position

        return (
            0 <= row < self.grid_size
            and
            0 <= col < self.grid_size
        )

    def _check_radar_detection(self):

        for radar in self.radars:

            if radar.detect(self.scout.position):

                radar.state = "TRACK"

                return True

        return False