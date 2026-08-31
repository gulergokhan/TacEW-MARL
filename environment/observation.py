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
        scout_id = scout.get("aircraft_id", "scout_01")
        hunter_x = hunter["x"]
        hunter_y = hunter["y"]
        hunter_id = hunter.get(
            "aircraft_id",
            "hunter_01",
        )

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
                        radar.state_for(scout_id)
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
                # Hunter radar state
                observation.extend(
                    self._radar_state_one_hot(
                        radar.state_for(hunter_id)
                    )
                )

                # Distance from Hunter to radar
                hunter_distance = np.sqrt(
                    (
                        hunter_x
                        - radar.position.x
                    ) ** 2
                    +
                    (
                        hunter_y
                        - radar.position.y
                    ) ** 2
                )

                observation.append(
                    self._normalize(
                        hunter_distance,
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
    # CTDE — CENTRALIZED TRAINING, DECENTRALIZED EXECUTION
    # ==================================================
    #
    # encode() above stays untouched (used as the legacy single-vector
    # observation by the existing single-agent DQN scripts).
    #
    # For MARL, each agent needs its own decentralized, ego-centric
    # observation for execution (encode_agent), while the centralized
    # critic during training gets the full joint state (encode_global,
    # currently an alias of encode() — kept as a separate name so critic
    # and policy inputs can diverge later without touching call sites).

    def encode_global(
        self,
        state: dict,
        terrain,
        radar_system=None,
    ) -> np.ndarray:
        """Full joint state for the centralized critic (CTDE)."""
        return self.encode(state, terrain, radar_system)

    def encode_agent(
        self,
        agent_id: str,
        state: dict,
        terrain,
        radar_system=None,
    ) -> np.ndarray:
        """
        Ego-centric, decentralized observation for one agent's policy
        ("scout" or "hunter"). Built from the agent's own frame: its own
        kinematics first, the other aircraft as a relative offset, local
        terrain around itself, and radar readings for itself only (no
        peeking at the teammate's radar tracks — that's the whole point
        of decentralized execution).
        """

        if agent_id not in ("scout", "hunter"):
            raise ValueError(f"Unknown agent_id: {agent_id}")

        other_id = "hunter" if agent_id == "scout" else "scout"

        self_state = state[agent_id]
        other_state = state[other_id]
        weather = state["weather"]

        self_x, self_y = self_state["x"], self_state["y"]
        self_agent_id = self_state.get(
            "aircraft_id",
            "scout_01" if agent_id == "scout" else "hunter_01",
        )

        observation = [
            # ---- own kinematics ----
            self._normalize_position(self_x, self.width),
            self._normalize_position(self_y, self.height),
            self._normalize(self_state["fuel"], 0.0, 100.0),
            self._normalize(self_state.get("heading", 0.0), 0.0, 360.0),

            # ---- teammate, relative to self (partial observability) ----
            self._normalize(
                (other_state["x"] - self_x) + self.width,
                0.0,
                2 * self.width,
            ),
            self._normalize(
                (other_state["y"] - self_y) + self.height,
                0.0,
                2 * self.height,
            ),
            self._normalize(other_state["fuel"], 0.0, 100.0),

            # ---- shared weather ----
            self._normalize(weather["temperature"], -50.0, 50.0),
            self._normalize(weather["wind_speed"], 0.0, 150.0),
            self._normalize(weather["wind_direction"], 0.0, 360.0),
            self._normalize(weather["visibility"], 0.0, 50000.0),
            self._normalize(weather["precipitation"], 0.0, 50.0),
            self._normalize(weather["cloud_cover"], 0.0, 100.0),
        ]

        # ---- local terrain 3x3 around self ----
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                terrain_type = self._get_terrain_type(
                    terrain, self_x + dx, self_y + dy,
                )
                observation.extend(self._terrain_one_hot(terrain_type))

        # ---- own radar picture only (decentralized: no teammate tracks) ----
        if radar_system is not None:
            for radar in radar_system.radars:

                observation.append(
                    self._normalize_position(radar.position.x, self.width)
                )
                observation.append(
                    self._normalize_position(radar.position.y, self.height)
                )
                observation.append(
                    self._normalize(
                        radar.detection_range, 0.0, max(self.width, self.height),
                    )
                )
                observation.append(1.0 if radar.active else 0.0)

                observation.extend(
                    self._radar_state_one_hot(radar.state_for(self_agent_id))
                )

                distance = np.sqrt(
                    (self_x - radar.position.x) ** 2
                    + (self_y - radar.position.y) ** 2
                )
                observation.append(
                    self._normalize(
                        distance, 0.0,
                        np.sqrt(self.width ** 2 + self.height ** 2),
                    )
                )

        return np.array(observation, dtype=np.float32)

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