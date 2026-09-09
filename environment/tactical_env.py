import json
from pathlib import Path

import requests

from aircraft.aircraft import Aircraft
from aircraft.models import AircraftType, Position

from weather.client import WeatherClient
from weather.models import WeatherState
from weather.parser import WeatherParser

from terrain.terrain_map import TerrainMap

from radar.models import Radar
from radar.radar import RadarSystem

from environment.observation import ObservationEncoder

from configs import environment_config as cfg


class TacticalEnv:

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_STAY = 4
    ACTION_JAM_SUPPRESS = 5
    ACTION_JAM_DECEIVE = 6

    NUM_ACTIONS = 7

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        latitude: float = 39.9334,
        longitude: float = 32.8597,
        weather: WeatherState | None = None,
        weather_client: WeatherClient | None = None,
        scout_start_position: tuple[int, int] | None = None,
        hunter_start_position: tuple[int, int] | None = None,
        strike_point: tuple[int, int] | None = None,
        terrain_cells: list[dict] | None = None,
        radars_config: list[dict] | None = None,
        cell_km: float = 4.0,
    ):
        """
        width/height, strike_point, terrain_cells and radars_config let a
        user-built scenario fully override the default 10x10 demo map.

        terrain_cells:
            [{"x": int, "y": int, "type": "MOUNTAIN"|"WATER"|
              "URBAN"|"FOREST"}, ...]

        radars_config:
            [{"radar_id": str, "x": int, "y": int,
              "detection_range": float}, ...]

        cell_km:
            Real-world kilometers per grid cell.
        """

        self.width = width
        self.height = height
        self.cell_km = cell_km

        self.latitude = latitude
        self.longitude = longitude

        self.scout_start_position = (
            scout_start_position
            or cfg.TACTICAL_SCOUT_START_POSITION
        )

        self.hunter_start_position = (
            hunter_start_position
            or cfg.TACTICAL_HUNTER_START_POSITION
        )

        self.max_steps = 100

        strike_x, strike_y = (
            strike_point or cfg.STRIKE_POINT
        )

        self.strike_point = Position(
            x=strike_x,
            y=strike_y,
        )

        self.terrain = TerrainMap(
            width=width,
            height=height,
        )

        if terrain_cells is not None:
            self._setup_custom_terrain(
                terrain_cells
            )
        else:
            self._setup_default_terrain()

        self.weather_client = (
            weather_client or WeatherClient()
        )

        self.weather = (
            weather or self._load_weather()
        )

        if radars_config is not None:

            self.radar_system = RadarSystem(
                radars=[
                    Radar(
                        radar_id=r["radar_id"],
                        position=Position(
                            x=r["x"],
                            y=r["y"],
                        ),
                        detection_range=r.get(
                            "detection_range",
                            3.0,
                        ),
                    )
                    for r in radars_config
                ]
            )

        else:

            self.radar_system = RadarSystem(
                radars=[
                    Radar(
                        radar_id="radar_01",
                        position=Position(
                            x=5,
                            y=5,
                        ),
                        detection_range=3.0,
                    ),
                    Radar(
                        radar_id="radar_02",
                        position=Position(
                            x=2,
                            y=7,
                        ),
                        detection_range=2.5,
                    ),
                    Radar(
                        radar_id="radar_03",
                        position=Position(
                            x=7,
                            y=3,
                        ),
                        detection_range=2.5,
                    ),
                ]
            )

        self.observation_encoder = ObservationEncoder(
            width=width,
            height=height,
        )

        self.scout = None
        self.hunter = None

        self.current_step = 0

        # --------------------------------------------------
        # MISSION STATE
        # --------------------------------------------------

        self.hunter_target_reached = False

        # --------------------------------------------------
        # DASHBOARD / EPISODE TRACE
        # --------------------------------------------------

        self.episode_log = []

    # ==================================================
    # SCENARIO
    # ==================================================

    @classmethod
    def from_scenario(
        cls,
        scenario: dict,
        **overrides,
    ) -> "TacticalEnv":

        cls._validate_scenario(scenario)

        kwargs = dict(
            width=scenario.get("width", 10),
            height=scenario.get("height", 10),
            cell_km=scenario.get("cell_km", 4.0),

            scout_start_position=(
                tuple(scenario["scout_start"])
                if "scout_start" in scenario
                else None
            ),

            hunter_start_position=(
                tuple(scenario["hunter_start"])
                if "hunter_start" in scenario
                else None
            ),

            strike_point=(
                tuple(scenario["strike_point"])
                if "strike_point" in scenario
                else None
            ),

            terrain_cells=scenario.get(
                "terrain_cells"
            ),

            radars_config=scenario.get(
                "radars"
            ),
        )

        kwargs.update(overrides)

        return cls(**kwargs)

    @staticmethod
    def _validate_scenario(
        scenario: dict,
    ):

        if not isinstance(
            scenario,
            dict,
        ):
            raise ValueError(
                "scenario must be a dictionary"
            )

        width = scenario.get(
            "width",
            10,
        )

        height = scenario.get(
            "height",
            10,
        )

        cell_km = scenario.get(
            "cell_km",
            4.0,
        )

        if type(width) is not int or width <= 0:

            raise ValueError(
                "width must be a positive integer"
            )

        if type(height) is not int or height <= 0:

            raise ValueError(
                "height must be a positive integer"
            )

        if (
            isinstance(
                cell_km,
                bool,
            )
            or not isinstance(
                cell_km,
                (int, float),
            )
            or cell_km <= 0
        ):

            raise ValueError(
                "cell_km must be a positive number"
            )

        # --------------------------------------------------
        # POSITIONS
        # --------------------------------------------------

        for field_name in (
            "scout_start",
            "hunter_start",
            "strike_point",
        ):

            if field_name not in scenario:
                continue

            position = scenario[field_name]

            if (
                not isinstance(
                    position,
                    (list, tuple),
                )
                or len(position) != 2
                or any(
                    type(coordinate) is not int
                    for coordinate in position
                )
            ):

                raise ValueError(
                    f"{field_name} must contain two integers"
                )

            x, y = position

            if not (
                0 <= x < width
                and 0 <= y < height
            ):

                raise ValueError(
                    f"{field_name} must be inside the grid"
                )

        # --------------------------------------------------
        # TERRAIN
        # --------------------------------------------------

        terrain_cells = scenario.get(
            "terrain_cells"
        )

        if terrain_cells is not None:

            if not isinstance(
                terrain_cells,
                list,
            ):

                raise ValueError(
                    "terrain_cells must be a list"
                )

            allowed_terrain_types = {
                "PLAIN",
                "MOUNTAIN",
                "WATER",
                "URBAN",
                "FOREST",
            }

            for cell in terrain_cells:

                if not isinstance(
                    cell,
                    dict,
                ):

                    raise ValueError(
                        "terrain_cells entries must be dictionaries"
                    )

                x = cell.get("x")
                y = cell.get("y")
                terrain_type = cell.get("type")

                if (
                    type(x) is not int
                    or type(y) is not int
                    or not (
                        0 <= x < width
                        and 0 <= y < height
                    )
                ):

                    raise ValueError(
                        "terrain_cells must be inside the grid"
                    )

                if (
                    terrain_type
                    not in allowed_terrain_types
                ):

                    raise ValueError(
                        "terrain_cells contains an invalid terrain type"
                    )

        # --------------------------------------------------
        # RADARS
        # --------------------------------------------------

        radars = scenario.get(
            "radars"
        )

        if radars is not None:

            if not isinstance(
                radars,
                list,
            ):

                raise ValueError(
                    "radars must be a list"
                )

            if (
                len(radars)
                > ObservationEncoder.MAX_RADARS
            ):

                raise ValueError(
                    "Maximum supported radar count is 3"
                )

            radar_ids = set()
            radar_positions = set()

            for radar in radars:

                if not isinstance(
                    radar,
                    dict,
                ):

                    raise ValueError(
                        "radars entries must be dictionaries"
                    )

                radar_id = radar.get(
                    "radar_id"
                )

                x = radar.get("x")
                y = radar.get("y")

                detection_range = radar.get(
                    "detection_range",
                    3.0,
                )

                if (
                    not isinstance(
                        radar_id,
                        str,
                    )
                    or not radar_id.strip()
                ):

                    raise ValueError(
                        "radar_id must be a non-empty string"
                    )

                if radar_id in radar_ids:

                    raise ValueError(
                        f"Duplicate radar_id: {radar_id}"
                    )

                radar_ids.add(
                    radar_id
                )

                if (
                    type(x) is not int
                    or type(y) is not int
                    or not (
                        0 <= x < width
                        and 0 <= y < height
                    )
                ):

                    raise ValueError(
                        "radars must be inside the grid"
                    )

                if (
                    x,
                    y,
                ) in radar_positions:

                    raise ValueError(
                        "radars cannot share the same position"
                    )

                radar_positions.add(
                    (x, y)
                )

                if (
                    isinstance(
                        detection_range,
                        bool,
                    )
                    or not isinstance(
                        detection_range,
                        (int, float),
                    )
                    or detection_range <= 0
                ):

                    raise ValueError(
                        "radar detection_range must be positive"
                    )

    # ==================================================
    # TERRAIN
    # ==================================================

    def _setup_custom_terrain(
        self,
        terrain_cells: list[dict],
    ):

        from terrain.models import TerrainType

        for cell in terrain_cells:

            terrain_type = getattr(
                TerrainType,
                cell["type"],
            )

            self.terrain.set_terrain(
                cell["x"],
                cell["y"],
                terrain_type,
            )

    def _setup_default_terrain(self):

        from terrain.models import TerrainType

        # Mountain ridge.
        for y in (
            3,
            4,
            5,
        ):

            self.terrain.set_terrain(
                6,
                y,
                TerrainType.MOUNTAIN,
            )

        # Forest patch.
        for x, y in [
            (3, 6),
            (4, 6),
            (3, 7),
        ]:

            self.terrain.set_terrain(
                x,
                y,
                TerrainType.FOREST,
            )

        # Water.
        for x, y in [
            (0, 5),
            (1, 5),
            (0, 6),
        ]:

            self.terrain.set_terrain(
                x,
                y,
                TerrainType.WATER,
            )

    # ==================================================
    # WEATHER
    # ==================================================

    def _load_weather(self):

        try:

            weather_data = (
                self.weather_client.get_weather(
                    latitude=self.latitude,
                    longitude=self.longitude,
                )
            )

            return WeatherParser.parse(
                weather_data
            )

        except (
            requests.RequestException,
            KeyError,
            TypeError,
            ValueError,
        ):

            return WeatherState(
                temperature=20.0,
                wind_speed=0.0,
                wind_direction=0.0,
                visibility=10000.0,
                precipitation=0.0,
                cloud_cover=0.0,
            )

    def _weather_dict(self):

        return {
            "temperature": self.weather.temperature,
            "wind_speed": self.weather.wind_speed,
            "wind_direction": self.weather.wind_direction,
            "visibility": self.weather.visibility,
            "precipitation": self.weather.precipitation,
            "cloud_cover": self.weather.cloud_cover,
        }

    # ==================================================
    # RESET
    # ==================================================

    def reset(self):

        self.current_step = 0

        self.episode_log = []

        self.hunter_target_reached = False

        self.scout = Aircraft(
            aircraft_id="scout_01",
            aircraft_type=AircraftType.SCOUT,
            start_x=self.scout_start_position[0],
            start_y=self.scout_start_position[1],
        )

        self.hunter = Aircraft(
            aircraft_id="hunter_01",
            aircraft_type=AircraftType.HUNTER,
            start_x=self.hunter_start_position[0],
            start_y=self.hunter_start_position[1],
        )

        self.radar_system.reset()

        state = self._get_state()

        observation = (
            self.observation_encoder.encode(
                state,
                self.terrain,
                self.radar_system,
            )
        )

        return observation

    # ==================================================
    # STEP
    # ==================================================

    def step(
        self,
        scout_action: int,
        hunter_action: int = None,
    ):
        """
        scout_action drives the Scout.

        hunter_action:
            None -> heuristic Hunter movement.
            int  -> explicit learned Hunter action.
        """

        self.current_step += 1

        self.radar_system.advance_time()

        # ==================================================
        # SCOUT
        # ==================================================

        reward, done, scout_info = (
            self._apply_scout_action(
                scout_action
            )
        )

        scout_reward = reward

        # ==================================================
        # HUNTER
        # ==================================================

        hunter_info = (
            self._apply_hunter_action(
                hunter_action
            )
        )

        hunter_reward = (
            self.hunter_reward
        )

        # ==================================================
        # HUNTER ACTION OUTCOME -> SHARED REWARD
        # ==================================================
        #
        # HAPPO learns from the shared team reward. Include
        # Hunter progress as well as invalid/jamming action
        # costs so those decisions cannot be hidden from
        # the cooperative learning signal.
        #

        hunter_progress_reward = (
            hunter_info["progress"]
            * cfg.HUNTER_PROGRESS_REWARD
        )

        reward += hunter_reward

        # ==================================================
        # ESCORT
        # ==================================================

        escort_distance = self._dist(
            self.scout.state.position.x,
            self.scout.state.position.y,
            self.hunter.state.position.x,
            self.hunter.state.position.y,
        )

        escort_in_range = (
            escort_distance
            <= cfg.ESCORT_RADIUS
        )

        if escort_in_range:

            reward += cfg.ESCORT_REWARD

            scout_reward += cfg.ESCORT_REWARD

            hunter_reward += cfg.ESCORT_REWARD

        else:

            escort_penalty = (
                escort_distance
                - cfg.ESCORT_RADIUS
            ) * cfg.ESCORT_DISTANCE_PENALTY

            reward -= escort_penalty

            scout_reward -= escort_penalty

            hunter_reward -= escort_penalty

        scout_info["escort_distance"] = (
            escort_distance
        )

        scout_info["escort_in_range"] = (
            escort_in_range
        )

        # ==================================================
        # RADAR DETECTION
        # ==================================================

        scout_detections = (
            self.radar_system.detect(
                self.scout.state.position,
                weather=self._weather_dict(),
                terrain=self.terrain,
                target_id=self.scout.state.aircraft_id,
            )
        )

        hunter_detections = (
            self.radar_system.detect(
                self.hunter.state.position,
                weather=self._weather_dict(),
                terrain=self.terrain,
                target_id=self.hunter.state.aircraft_id,
            )
        )

        scout_lethal_hit = any(
            detection["state"].value == "LETHAL"
            for detection in scout_detections
        )

        hunter_lethal_hit = any(
            detection["state"].value == "LETHAL"
            for detection in hunter_detections
        )

        # ==================================================
        # SCOUT DETECTION PENALTY
        # ==================================================

        if scout_detections:

            detection_penalty = float(
                len(scout_detections)
                * cfg.SCOUT_DETECTION_PENALTY
            )

            reward -= detection_penalty

            scout_reward -= detection_penalty

        # ==================================================
        # HUNTER DETECTION PENALTY
        # ==================================================

        if hunter_detections:

            hunter_detection_penalty = float(
                len(hunter_detections)
                * cfg.HUNTER_DETECTION_PENALTY
            )

            reward -= hunter_detection_penalty

            hunter_reward -= hunter_detection_penalty

        # ==================================================
        # HUNTER GOAL REWARD
        # ==================================================

        if (
            hunter_info[
                "target_reached_this_step"
            ]
            and not hunter_lethal_hit
        ):

            reward += cfg.HUNTER_GOAL_REWARD

            hunter_reward += (
                cfg.HUNTER_GOAL_REWARD
            )

        # ==================================================
        # TERMINATION CONDITIONS
        # ==================================================

        radar_failure = (
            scout_lethal_hit
            or hunter_lethal_hit
        )

        fuel_exhausted = (
            self.scout.state.fuel <= 0
            or self.hunter.state.fuel <= 0
        )

        time_limit_reached = (
            self.current_step
            >= self.max_steps
        )

        mission_success = (
            self.hunter_target_reached
            and escort_in_range
            and not radar_failure
        )

        # --------------------------------------------------
        # CRITICAL:
        # Hunter reaching strike point is itself a
        # terminal condition.
        #
        # This prevents:
        #
        # (1,8) -> (0,8) -> ...
        #
        # after the Hunter has already reached the target.
        # --------------------------------------------------

        hunter_reached_target = (
            hunter_info[
                "target_reached_this_step"
            ]
        )

        done = False

        termination_reason = None

        # ==================================================
        # RADAR FAILURE
        # ==================================================

        if radar_failure:

            if scout_lethal_hit:

                reward += cfg.LETHAL_PENALTY

                scout_reward += (
                    cfg.LETHAL_PENALTY
                )

            if hunter_lethal_hit:

                reward += (
                    cfg.HUNTER_LETHAL_PENALTY
                )

                hunter_reward += (
                    cfg.HUNTER_LETHAL_PENALTY
                )

            done = True

            if (
                scout_lethal_hit
                and hunter_lethal_hit
            ):

                termination_reason = (
                    "both_lethal"
                )

            elif scout_lethal_hit:

                termination_reason = (
                    "scout_lethal"
                )

            else:

                termination_reason = (
                    "hunter_lethal"
                )

        # ==================================================
        # MISSION SUCCESS
        # ==================================================

        elif mission_success:

            reward += (
                cfg.MISSION_SUCCESS_REWARD
            )

            scout_reward += (
                cfg.MISSION_SUCCESS_REWARD
            )

            hunter_reward += (
                cfg.MISSION_SUCCESS_REWARD
            )

            done = True

            termination_reason = (
                "mission_success"
            )

        # ==================================================
        # HUNTER REACHED STRIKE POINT
        # ==================================================

        elif hunter_reached_target:

            reward += (
                cfg.UNESCORTED_TARGET_PENALTY
            )

            scout_reward += (
                cfg.UNESCORTED_TARGET_PENALTY
            )

            hunter_reward += (
                cfg.UNESCORTED_TARGET_PENALTY
            )

            done = True

            termination_reason = (
                "hunter_reached_target"
            )
        # ==================================================
        # FUEL
        # ==================================================

        elif fuel_exhausted:

            reward += (
                cfg.FUEL_EXHAUSTED_PENALTY
            )

            scout_reward += (
                cfg.FUEL_EXHAUSTED_PENALTY
            )

            hunter_reward += (
                cfg.FUEL_EXHAUSTED_PENALTY
            )

            done = True

            termination_reason = (
                "fuel_exhausted"
            )

        # ==================================================
        # TIME LIMIT
        # ==================================================

        elif time_limit_reached:

            reward += (
                cfg.TIME_LIMIT_PENALTY
            )

            scout_reward += (
                cfg.TIME_LIMIT_PENALTY
            )

            hunter_reward += (
                cfg.TIME_LIMIT_PENALTY
            )

            done = True

            termination_reason = (
                "time_limit"
            )

        mission_failed = (
            done
            and not mission_success
        )

        # ==================================================
        # RADAR STATUS
        # ==================================================

        radar_status = (
            self.radar_system.get_status(
                self.scout.state.position,
                weather=self._weather_dict(),
                terrain=self.terrain,
                target_id=self.scout.state.aircraft_id,
            )
        )

        radar_status.extend(
            self.radar_system.get_status(
                self.hunter.state.position,
                weather=self._weather_dict(),
                terrain=self.terrain,
                target_id=self.hunter.state.aircraft_id,
            )
        )

        # ==================================================
        # SCOUT INFO
        # ==================================================

        scout_info.update(
            {
                "radar_detections": [
                    {
                        **detection,
                        "state": detection[
                            "state"
                        ].value,
                    }
                    for detection
                    in scout_detections
                ],

                "hunter_radar_detections": [
                    {
                        **detection,
                        "state": detection[
                            "state"
                        ].value,
                    }
                    for detection
                    in hunter_detections
                ],

                "radar_status": radar_status,

                "lethal_hit": (
                    scout_lethal_hit
                ),

                "hunter_lethal_hit": (
                    hunter_lethal_hit
                ),

                "mission_success": (
                    mission_success
                ),

                "mission_failed": (
                    mission_failed
                ),

                "termination_reason": (
                    termination_reason
                ),
            }
        )

        hunter_info["lethal_hit"] = (
            hunter_lethal_hit
        )

        # Explicitly expose progress reward
        # for diagnostics/dashboard.
        hunter_info["progress_reward"] = (
            hunter_progress_reward
        )

        # ==================================================
        # OBSERVATION
        # ==================================================

        state = self._get_state()

        observation = (
            self.observation_encoder.encode(
                state,
                self.terrain,
                self.radar_system,
            )
        )

        # ==================================================
        # CTDE OBSERVATIONS
        # ==================================================

        agent_observations = (
            self._get_agent_observations(
                state
            )
        )

        # ==================================================
        # FINAL INFO
        # ==================================================

        info = {
            **scout_info,

            "hunter": hunter_info,

            "step": self.current_step,

            "scout_action": scout_action,

            "hunter_action": hunter_action,

            "reward": reward,

            "reward_scout": scout_reward,

            "reward_hunter": hunter_reward,

            "hunter_progress_reward": (
                hunter_progress_reward
            ),

            "weather": self._weather_dict(),

            "observations": {
                "scout": agent_observations[
                    "scout"
                ],

                "hunter": agent_observations[
                    "hunter"
                ],
            },

            "global_state": agent_observations[
                "global"
            ],

            "heading": {
                "scout": (
                    self.scout.state.heading
                ),

                "hunter": (
                    self.hunter.state.heading
                ),
            },
        }

        self._log_step(info)

        return (
            observation,
            reward,
            done,
            info,
        )

    # ==================================================
    # SCOUT ACTION
    # ==================================================

    def _apply_scout_action(
        self,
        action: int,
    ):

        old_x = (
            self.scout.state.position.x
        )

        old_y = (
            self.scout.state.position.y
        )

        reward = -0.05
        done = False

        # --------------------------------------------------
        # MOVEMENT
        # --------------------------------------------------

        if action in (
            self.ACTION_UP,
            self.ACTION_DOWN,
            self.ACTION_LEFT,
            self.ACTION_RIGHT,
            self.ACTION_STAY,
        ):

            new_x = old_x
            new_y = old_y

            if action == self.ACTION_UP:

                new_y -= 1

            elif action == self.ACTION_DOWN:

                new_y += 1

            elif action == self.ACTION_LEFT:

                new_x -= 1

            elif action == self.ACTION_RIGHT:

                new_x += 1

            if action == self.ACTION_STAY:

                reward -= 0.15

            movement_valid = True

            if not self._is_inside_grid(
                new_x,
                new_y,
            ):

                reward -= 1.0

                movement_valid = False

            elif not self.terrain.is_passable(
                new_x,
                new_y,
            ):

                reward -= 1.0

                movement_valid = False

            if movement_valid:

                self.scout.move(
                    new_x,
                    new_y,
                )

                self.scout.consume_fuel(
                    1.0
                )

        # --------------------------------------------------
        # JAMMING
        # --------------------------------------------------

        elif action in (
            self.ACTION_JAM_SUPPRESS,
            self.ACTION_JAM_DECEIVE,
        ):

            self.scout.consume_fuel(
                1.5
            )

            target_radar = (
                self.radar_system.nearest_radar(
                    self.scout.state.position,
                    max_range=cfg.JAM_RANGE,
                    exclude_suppressed=(
                        action
                        == self.ACTION_JAM_SUPPRESS
                    ),
                    exclude_deceived=(
                        action
                        == self.ACTION_JAM_DECEIVE
                    ),
                )
            )

            if target_radar is None:
                target_radar = (
                    self.radar_system.nearest_radar(
                        self.scout.state.position,
                        max_range=cfg.JAM_RANGE,
                    )
                )

            if target_radar is None:

                reward -= 0.5

            elif (
                action
                == self.ACTION_JAM_SUPPRESS
            ):

                if (
                    target_radar.suppression_timer
                    > 0
                ):

                    reward -= 0.25

                else:

                    self.radar_system.jam_suppress(
                        target_radar.radar_id,
                        strength=(
                            cfg.JAM_SUPPRESSION_STRENGTH
                        ),
                        duration=(
                            cfg.JAM_SUPPRESSION_DURATION
                        ),
                    )

                    reward += 0.2

            else:

                if (
                    target_radar.deception_timer
                    > 0
                ):

                    reward -= 0.25

                else:

                    self.radar_system.jam_deceive(
                        target_radar.radar_id,
                        duration=(
                            cfg.JAM_DECEPTION_DURATION
                        ),
                    )

                    reward += 0.2

        else:

            raise ValueError(
                f"Invalid action: {action}"
            )

        # --------------------------------------------------
        # INFO
        # --------------------------------------------------

        new_x = (
            self.scout.state.position.x
        )

        new_y = (
            self.scout.state.position.y
        )

        info = {
            "scout": {
                "x": new_x,
                "y": new_y,
                "fuel": self.scout.state.fuel,
                "heading": self.scout.state.heading,
            },

            "goal": {
                "x": self.strike_point.x,
                "y": self.strike_point.y,
            },

            "action": self.scout_action_name(
                action
            ),
        }

        return (
            reward,
            done,
            info,
        )

    # ==================================================
    # HUNTER ACTION
    # ==================================================

    def _apply_hunter_action(
        self,
        hunter_action: int | None,
    ):

        self.hunter_reward = 0.0

        # --------------------------------------------------
        # DISTANCE BEFORE ACTION
        # --------------------------------------------------

        previous_distance = self._dist(
            self.hunter.state.position.x,
            self.hunter.state.position.y,
            self.strike_point.x,
            self.strike_point.y,
        )

        # --------------------------------------------------
        # ACTION
        # --------------------------------------------------

        if hunter_action is None:

            self._hunter_heuristic_move()

        else:

            self._hunter_apply_explicit_action(
                hunter_action
            )

        # --------------------------------------------------
        # CURRENT POSITION
        # --------------------------------------------------

        hx = self.hunter.state.position.x
        hy = self.hunter.state.position.y

        # --------------------------------------------------
        # DISTANCE AFTER ACTION
        # --------------------------------------------------

        current_distance = self._dist(
            hx,
            hy,
            self.strike_point.x,
            self.strike_point.y,
        )

        # ==================================================
        # PROGRESS
        # ==================================================
        #
        # Positive:
        #   Hunter moved toward target.
        #
        # Negative:
        #   Hunter moved away from target.
        #
        # Zero:
        #   Hunter did not change distance.
        #

        progress = (
            previous_distance
            - current_distance
        )

        progress_reward = (
            progress
            * cfg.HUNTER_PROGRESS_REWARD
        )

        self.hunter_reward += (
            progress_reward
        )

        # ==================================================
        # TARGET REACHED
        # ==================================================

        hunter_reached_target_now = (
            hx == self.strike_point.x
            and hy == self.strike_point.y
        )

        target_reached_this_step = (
            hunter_reached_target_now
            and not self.hunter_target_reached
        )

        if target_reached_this_step:

            self.hunter_target_reached = True

        # ==================================================
        # INFO
        # ==================================================

        return {
            "x": hx,

            "y": hy,

            "fuel": self.hunter.state.fuel,

            "heading": self.hunter.state.heading,

            "strike_point": {
                "x": self.strike_point.x,
                "y": self.strike_point.y,
            },

            "target_reached": (
                self.hunter_target_reached
            ),

            "target_reached_this_step": (
                target_reached_this_step
            ),

            "progress": progress,

            "progress_reward": (
                progress_reward
            ),
        }

    # ==================================================
    # HUNTER HEURISTIC
    # ==================================================

    def _hunter_heuristic_move(self):
        """
        Greedy pursuit toward the strike point.
        """

        if not self.hunter.is_operational():

            return

        hx = self.hunter.state.position.x
        hy = self.hunter.state.position.y

        if (
            hx == self.strike_point.x
            and hy == self.strike_point.y
        ):

            return

        candidates = [
            (hx, hy - 1),
            (hx, hy + 1),
            (hx - 1, hy),
            (hx + 1, hy),
        ]

        radar_cells = {
            (
                radar.position.x,
                radar.position.y,
            )
            for radar in self.radar_system.radars
            if radar.active
        }

        current_distance = self._dist(
            hx,
            hy,
            self.strike_point.x,
            self.strike_point.y,
        )

        best = None
        best_distance = current_distance

        for cx, cy in candidates:

            if not self._is_inside_grid(
                cx,
                cy,
            ):

                continue

            if not self.terrain.is_passable(
                cx,
                cy,
            ):

                continue

            if (
                cx,
                cy,
            ) in radar_cells:

                continue

            distance = self._dist(
                cx,
                cy,
                self.strike_point.x,
                self.strike_point.y,
            )

            if distance < best_distance:

                best_distance = distance

                best = (
                    cx,
                    cy,
                )

        if best is not None:

            self.hunter.move(
                *best
            )

            self.hunter.consume_fuel(
                1.0
            )

    # ==================================================
    # HUNTER EXPLICIT ACTION
    # ==================================================

    def _hunter_apply_explicit_action(
        self,
        action: int,
    ):
        """
        Explicit action for the learned Hunter policy.

        HAPPO action space:
            0 UP
            1 DOWN
            2 LEFT
            3 RIGHT
            4 STAY

        Legacy direct callers may still use:
            5 JAM_SUPPRESS
            6 JAM_DECEIVE
        """

        self.hunter_reward = 0.0

        if not self.hunter.is_operational():

            return

        hx = self.hunter.state.position.x
        hy = self.hunter.state.position.y

        # ==================================================
        # MOVEMENT
        # ==================================================

        if action in (
            self.ACTION_UP,
            self.ACTION_DOWN,
            self.ACTION_LEFT,
            self.ACTION_RIGHT,
            self.ACTION_STAY,
        ):

            new_x = hx
            new_y = hy

            if action == self.ACTION_UP:

                new_y -= 1

            elif action == self.ACTION_DOWN:

                new_y += 1

            elif action == self.ACTION_LEFT:

                new_x -= 1

            elif action == self.ACTION_RIGHT:

                new_x += 1

            if action == self.ACTION_STAY:

                self.hunter_reward -= 0.15

            if (
                self._is_inside_grid(
                    new_x,
                    new_y,
                )
                and self.terrain.is_passable(
                    new_x,
                    new_y,
                )
            ):

                self.hunter.move(
                    new_x,
                    new_y,
                )

                self.hunter.consume_fuel(
                    1.0
                )

            else:

                self.hunter_reward -= 1.0

        # ==================================================
        # JAMMING
        # ==================================================

        elif action in (
            self.ACTION_JAM_SUPPRESS,
            self.ACTION_JAM_DECEIVE,
        ):

            self.hunter.consume_fuel(
                1.5
            )

            target_radar = (
                self.radar_system.nearest_radar(
                    self.hunter.state.position,
                    max_range=cfg.JAM_RANGE,
                    exclude_suppressed=(
                        action
                        == self.ACTION_JAM_SUPPRESS
                    ),
                    exclude_deceived=(
                        action
                        == self.ACTION_JAM_DECEIVE
                    ),
                )
            )

            if target_radar is None:
                target_radar = (
                    self.radar_system.nearest_radar(
                        self.hunter.state.position,
                        max_range=cfg.JAM_RANGE,
                    )
                )

            if target_radar is None:

                self.hunter_reward -= 0.5

            elif (
                action
                == self.ACTION_JAM_SUPPRESS
            ):

                if (
                    target_radar.suppression_timer
                    > 0
                ):

                    self.hunter_reward -= 0.25

                else:

                    self.radar_system.jam_suppress(
                        target_radar.radar_id,
                        strength=(
                            cfg.JAM_SUPPRESSION_STRENGTH
                        ),
                        duration=(
                            cfg.JAM_SUPPRESSION_DURATION
                        ),
                    )

                    self.hunter_reward += 0.2

            else:

                if (
                    target_radar.deception_timer
                    > 0
                ):

                    self.hunter_reward -= 0.25

                else:

                    self.radar_system.jam_deceive(
                        target_radar.radar_id,
                        duration=(
                            cfg.JAM_DECEPTION_DURATION
                        ),
                    )

                    self.hunter_reward += 0.2

        else:

            raise ValueError(
                f"Invalid Hunter action: {action}"
            )

    # ==================================================
    # HELPERS
    # ==================================================

    def scout_action_name(
        self,
        action: int,
    ) -> str:

        names = {
            self.ACTION_UP: "UP",
            self.ACTION_DOWN: "DOWN",
            self.ACTION_LEFT: "LEFT",
            self.ACTION_RIGHT: "RIGHT",
            self.ACTION_STAY: "STAY",
            self.ACTION_JAM_SUPPRESS: (
                "JAM_SUPPRESS"
            ),
            self.ACTION_JAM_DECEIVE: (
                "JAM_DECEIVE"
            ),
        }

        return names.get(
            action,
            "UNKNOWN",
        )

    @staticmethod
    def _dist(
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> float:

        return (
            (x0 - x1) ** 2
            + (y0 - y1) ** 2
        ) ** 0.5

    def _is_inside_grid(
        self,
        x: int,
        y: int,
    ) -> bool:

        return (
            0 <= x < self.width
            and 0 <= y < self.height
        )

    # ==================================================
    # STATE
    # ==================================================

    def _get_state(self):

        return {
            "scout": {
                "aircraft_id": (
                    self.scout.state.aircraft_id
                ),

                "x": (
                    self.scout.state.position.x
                ),

                "y": (
                    self.scout.state.position.y
                ),

                "fuel": (
                    self.scout.state.fuel
                ),

                "heading": (
                    self.scout.state.heading
                ),
            },

            "hunter": {
                "aircraft_id": (
                    self.hunter.state.aircraft_id
                ),

                "x": (
                    self.hunter.state.position.x
                ),

                "y": (
                    self.hunter.state.position.y
                ),

                "fuel": (
                    self.hunter.state.fuel
                ),

                "heading": (
                    self.hunter.state.heading
                ),
            },

            "strike_point": {
                "x": self.strike_point.x,
                "y": self.strike_point.y,
            },

            "weather": self._weather_dict(),
        }

    # ==================================================
    # CTDE OBSERVATIONS
    # ==================================================

    def _get_agent_observations(
        self,
        state: dict,
    ):

        return {
            "scout": (
                self.observation_encoder.encode_agent(
                    "scout",
                    state,
                    self.terrain,
                    self.radar_system,
                )
            ),

            "hunter": (
                self.observation_encoder.encode_agent(
                    "hunter",
                    state,
                    self.terrain,
                    self.radar_system,
                )
            ),

            "global": (
                self.observation_encoder.encode_global(
                    state,
                    self.terrain,
                    self.radar_system,
                )
            ),
        }

    # ==================================================
    # DASHBOARD LOGGING
    # ==================================================

    def _log_step(
        self,
        info: dict,
    ):

        terrain_grid = None

        if self.current_step == 1:

            terrain_grid = (
                self.terrain.grid.tolist()
            )

        entry = {
            "step": self.current_step,

            "scout": info["scout"],

            "hunter": info["hunter"],

            "goal": info["goal"],

            "radar_status": info[
                "radar_status"
            ],

            "weather": info["weather"],

            "reward": info["reward"],

            "reward_hunter": info.get(
                "reward_hunter"
            ),

            "hunter_progress_reward": (
                info.get(
                    "hunter_progress_reward"
                )
            ),

            "action": info["action"],

            "hunter_action": info[
                "hunter_action"
            ],

            "lethal_hit": info[
                "lethal_hit"
            ],

            "hunter_lethal_hit": info[
                "hunter_lethal_hit"
            ],

            "escort_distance": info[
                "escort_distance"
            ],

            "escort_in_range": info[
                "escort_in_range"
            ],

            "mission_success": info[
                "mission_success"
            ],

            "mission_failed": info[
                "mission_failed"
            ],

            "termination_reason": info[
                "termination_reason"
            ],
        }

        if terrain_grid is not None:

            entry["terrain_grid"] = (
                terrain_grid
            )

            entry["cell_km"] = (
                self.cell_km
            )

            entry["radars"] = [
                {
                    "radar_id": (
                        radar.radar_id
                    ),

                    "x": radar.position.x,

                    "y": radar.position.y,

                    "detection_range": (
                        radar.detection_range
                    ),
                }

                for radar
                in self.radar_system.radars
            ]

            entry["strike_point"] = {
                "x": self.strike_point.x,
                "y": self.strike_point.y,
            }

        self.episode_log.append(
            entry
        )

    # ==================================================
    # EXPORT
    # ==================================================

    def export_log(
        self,
        path: str | Path,
    ):

        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.episode_log,
                file,
                indent=2,
            )
