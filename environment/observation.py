import numpy as np

from terrain.models import TerrainType


class ObservationEncoder:

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def encode(self, state: dict, terrain) -> np.ndarray:
        scout = state["scout"]
        hunter = state["hunter"]
        weather = state["weather"]

        observation = [
            # Scout
            self._normalize_position(
                scout["x"],
                self.width,
            ),
            self._normalize_position(
                scout["y"],
                self.height,
            ),
            self._normalize(
                scout["fuel"],
                0.0,
                100.0,
            ),

            # Hunter
            self._normalize_position(
                hunter["x"],
                self.width,
            ),
            self._normalize_position(
                hunter["y"],
                self.height,
            ),
            self._normalize(
                hunter["fuel"],
                0.0,
                100.0,
            ),

            # Weather
            self._normalize(
                weather["temperature"],
                -50.0,
                50.0,
            ),
            self._normalize(
                weather["wind_speed"],
                0.0,
                150.0,
            ),
            self._normalize(
                weather["wind_direction"],
                0.0,
                360.0,
            ),
            self._normalize(
                weather["visibility"],
                0.0,
                50000.0,
            ),
            self._normalize(
                weather["precipitation"],
                0.0,
                50.0,
            ),
            self._normalize(
                weather["cloud_cover"],
                0.0,
                100.0,
            ),
        ]

        # 3x3 local terrain around Scout
        scout_x = scout["x"]
        scout_y = scout["y"]

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):

                x = scout_x + dx
                y = scout_y + dy

                terrain_type = self._get_terrain_type(
                    terrain,
                    x,
                    y,
                )

                observation.extend(
                    self._terrain_one_hot(terrain_type)
                )

        return np.array(
            observation,
            dtype=np.float32,
        )

    def _get_terrain_type(
        self,
        terrain,
        x: int,
        y: int,
    ):
        if not (
            0 <= x < self.width
            and 0 <= y < self.height
        ):
            return None

        return terrain.get_terrain(x, y)

    @staticmethod
    def _terrain_one_hot(terrain_type):
        values = [0.0] * len(TerrainType)

        if terrain_type is None:
            return values

        index = int(terrain_type)
        values[index] = 1.0

        return values

    @staticmethod
    def _normalize(
        value: float,
        min_value: float,
        max_value: float,
    ) -> float:
        normalized = (
            (value - min_value)
            / (max_value - min_value)
        )

        return float(
            np.clip(normalized, 0.0, 1.0)
        )

    @staticmethod
    def _normalize_position(
        value: int,
        size: int,
    ) -> float:
        if size <= 1:
            return 0.0

        return value / (size - 1)