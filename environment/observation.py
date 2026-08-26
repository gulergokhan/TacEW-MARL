import numpy as np

from terrain.models import TerrainType
from radar.models import RadarState


class ObservationEncoder:

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def encode(
        self,
        state: dict,
        terrain,
        radar_system=None,
    ) -> np.ndarray:

        scout = state["scout"]
        hunter = state["hunter"]
        weather = state["weather"]

        observation = [

            # ==========================================
            # SCOUT
            # ==========================================

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

            # ==========================================
            # HUNTER
            # ==========================================

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

            # ==========================================
            # WEATHER
            # ==========================================

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

        # ==========================================
        # LOCAL TERRAIN 3x3
        # ==========================================

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
                    self._terrain_one_hot(
                        terrain_type
                    )
                )

        # ==========================================
        # RADAR INFORMATION
        # ==========================================

        if radar_system is not None:

            for radar in radar_system.radars:

                # Radar position
                observation.append(
                    self._normalize_position(
                        radar.position.x,
                        self.width,
                    )
                )

                observation.append(
                    self._normalize_position(
                        radar.position.y,
                        self.height,
                    )
                )

                # Detection range
                observation.append(
                    self._normalize(
                        radar.detection_range,
                        0.0,
                        max(self.width, self.height),
                    )
                )

                # Radar active
                observation.append(
                    1.0 if radar.active else 0.0
                )

                # Radar state one-hot
                observation.extend(
                    self._radar_state_one_hot(
                        radar.state
                    )
                )

                # Distance from Scout to Radar
                distance = np.sqrt(
                    (
                        scout_x
                        - radar.position.x
                    ) ** 2
                    +
                    (
                        scout_y
                        - radar.position.y
                    ) ** 2
                )

                observation.append(
                    self._normalize(
                        distance,
                        0.0,
                        np.sqrt(
                            self.width ** 2
                            + self.height ** 2
                        ),
                    )
                )

        return np.array(
            observation,
            dtype=np.float32,
        )

    # ==================================================
    # TERRAIN
    # ==================================================

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

        return terrain.get_terrain(
            x,
            y,
        )

    @staticmethod
    def _terrain_one_hot(
        terrain_type
    ):

        values = [
            0.0
        ] * len(TerrainType)

        if terrain_type is None:
            return values

        index = int(terrain_type)

        values[index] = 1.0

        return values

    # ==================================================
    # RADAR STATE
    # ==================================================

    @staticmethod
    def _radar_state_one_hot(
        radar_state
    ):

        states = [
            RadarState.SAFE,
            RadarState.TRACK,
            RadarState.LOCK,
            RadarState.LETHAL,
        ]

        values = [
            0.0
        ] * len(states)

        if radar_state in states:

            index = states.index(
                radar_state
            )

            values[index] = 1.0

        return values

    # ==================================================
    # NORMALIZATION
    # ==================================================

    @staticmethod
    def _normalize(
        value: float,
        min_value: float,
        max_value: float,
    ) -> float:

        if max_value <= min_value:
            return 0.0

        normalized = (
            (value - min_value)
            /
            (max_value - min_value)
        )

        return float(
            np.clip(
                normalized,
                0.0,
                1.0,
            )
        )

    # ==================================================
    # POSITION
    # ==================================================

    @staticmethod
    def _normalize_position(
        value: int,
        size: int,
    ) -> float:

        if size <= 1:
            return 0.0

        return value / (size - 1)