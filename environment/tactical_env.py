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
    ):
        self.width = width
        self.height = height

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

        self.strike_point = Position(
            x=cfg.STRIKE_POINT[0],
            y=cfg.STRIKE_POINT[1],
        )

        self.terrain = TerrainMap(
            width=width,
            height=height,
        )

        # A modest terrain layout so line-of-sight/passability actually matter.
        self._setup_default_terrain()

        self.weather_client = weather_client or WeatherClient()
        self.weather = weather or self._load_weather()

        self.radar_system = RadarSystem(
            radars=[
                Radar(
                    radar_id="radar_01",
                    position=Position(x=5, y=5),
                    detection_range=3.0,
                ),
                Radar(
                    radar_id="radar_02",
                    position=Position(x=2, y=7),
                    detection_range=2.5,
                ),
                Radar(
                    radar_id="radar_03",
                    position=Position(x=7, y=3),
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

        # Per-step trace used to feed the dashboard. Populated on reset()/step().
        self.episode_log = []

    def _setup_default_terrain(self):

        from terrain.models import TerrainType

        # A mountain ridge that creates radar shadow between radar_01 and
        # the strike point, plus a small forest patch and a lake.
        for y in (3, 4, 5):
            self.terrain.set_terrain(6, y, TerrainType.MOUNTAIN)

        for x, y in [(3, 6), (4, 6), (3, 7)]:
            self.terrain.set_terrain(x, y, TerrainType.FOREST)

        for x, y in [(0, 5), (1, 5), (0, 6)]:
            self.terrain.set_terrain(x, y, TerrainType.WATER)

    def _load_weather(self):
        try:
            weather_data = self.weather_client.get_weather(
                latitude=self.latitude,
                longitude=self.longitude,
            )
            return WeatherParser.parse(weather_data)
        except (requests.RequestException, KeyError, TypeError, ValueError):
            # Keep training and tests available when the weather API is down.
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

        observation = self.observation_encoder.encode(
            state,
            self.terrain,
            self.radar_system,
        )

        return observation

    # ==================================================
    # STEP
    # ==================================================

    def step(self, scout_action: int, hunter_action: int = None):
        """
        scout_action drives the Scout (movement or jamming).
        hunter_action is optional: pass None to let the Hunter follow its
        built-in pursuit heuristic toward the strike point (used until the
        Hunter becomes its own learning agent under MARL); pass an explicit
        action to control it directly.
        """

        self.current_step += 1
        self.radar_system.advance_time()

        reward, done, scout_info = self._apply_scout_action(scout_action)

        hunter_info = self._apply_hunter_action(hunter_action)

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
        else:
            reward -= (
                escort_distance
                - cfg.ESCORT_RADIUS
            ) * cfg.ESCORT_DISTANCE_PENALTY

        scout_info["escort_distance"] = escort_distance
        scout_info["escort_in_range"] = escort_in_range

        scout_detections = self.radar_system.detect(
            self.scout.state.position,
            weather=self._weather_dict(),
            terrain=self.terrain,
            target_id=self.scout.state.aircraft_id,
        )
        hunter_detections = self.radar_system.detect(
            self.hunter.state.position,
            weather=self._weather_dict(),
            terrain=self.terrain,
            target_id=self.hunter.state.aircraft_id,
        )

        scout_lethal_hit = any(
            detection["state"].value == "LETHAL" for detection in scout_detections
        )
        hunter_lethal_hit = any(
            detection["state"].value == "LETHAL" for detection in hunter_detections
        )

        if scout_detections:
            reward -= float(len(scout_detections)*cfg.SCOUT_DETECTION_PENALTY)

        if hunter_detections:
            reward -= float(len(hunter_detections)*cfg.HUNTER_DETECTION_PENALTY)

        if hunter_info["target_reached_this_step"] and not hunter_lethal_hit:
            reward += cfg.HUNTER_GOAL_REWARD

        radar_failure = scout_lethal_hit or hunter_lethal_hit

        fuel_exhausted = self.scout.state.fuel <= 0 or self.hunter.state.fuel <= 0

        time_limit_reached = self.current_step >= self.max_steps

        mission_success = (
            self.hunter_target_reached and escort_in_range and not radar_failure
        )

        done = False
        termination_reason = None

        if radar_failure:

            if scout_lethal_hit:
                reward += cfg.LETHAL_PENALTY

            if hunter_lethal_hit:
                reward += cfg.HUNTER_LETHAL_PENALTY

            done = True

            if scout_lethal_hit and hunter_lethal_hit:
                termination_reason = "both_lethal"
            elif scout_lethal_hit:
                termination_reason = "scout_lethal"
            else:
                termination_reason = "hunter_lethal"

        elif mission_success:

            reward += cfg.MISSION_SUCCESS_REWARD
            done = True
            termination_reason = "mission_success"

        elif fuel_exhausted:

            done = True
            termination_reason = "fuel_exhausted"

        elif time_limit_reached:

            done = True
            termination_reason = "time_limit"

        mission_failed = done and not mission_success

        radar_status = self.radar_system.get_status(
            self.scout.state.position,
            weather=self._weather_dict(),
            terrain=self.terrain,
            target_id=self.scout.state.aircraft_id,
        )

        radar_status.extend(
            self.radar_system.get_status(
                self.hunter.state.position,
                weather=self._weather_dict(),
                terrain=self.terrain,
                target_id=self.hunter.state.aircraft_id,
            )
        )

        scout_info.update(
            {
                "radar_detections": [
                    {**detection, "state": detection["state"].value}
                    for detection in scout_detections
                ],
                "hunter_radar_detections": [
                    {**detection, "state": detection["state"].value}
                    for detection in hunter_detections
                ],
                "radar_status": radar_status,
                "lethal_hit": scout_lethal_hit,
                "hunter_lethal_hit": hunter_lethal_hit,
                "mission_success": mission_success,
                "mission_failed": mission_failed,
                "termination_reason": termination_reason,
            }
        )

        hunter_info["lethal_hit"] = hunter_lethal_hit

        state = self._get_state()

        observation = self.observation_encoder.encode(
            state,
            self.terrain,
            self.radar_system,
        )

        info = {
            **scout_info,
            "hunter": hunter_info,
            "step": self.current_step,
            "scout_action": scout_action,
            "hunter_action": hunter_action,
            "reward": reward,
            "weather": self._weather_dict(),
        }

        self._log_step(info)

        return (
            observation,
            reward,
            done,
            info,
        )

    def _apply_scout_action(self, action: int):

        old_x = self.scout.state.position.x
        old_y = self.scout.state.position.y

        reward = -0.05
        done = False

        if action in (
            self.ACTION_UP,
            self.ACTION_DOWN,
            self.ACTION_LEFT,
            self.ACTION_RIGHT,
            self.ACTION_STAY,
        ):

            new_x, new_y = old_x, old_y

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

            if not self._is_inside_grid(new_x, new_y):
                reward -= 1.0
                movement_valid = False

            elif not self.terrain.is_passable(new_x, new_y):
                reward -= 1.0
                movement_valid = False

            if movement_valid:
                self.scout.move(new_x, new_y)
                self.scout.consume_fuel(1.0)

        elif action in (self.ACTION_JAM_SUPPRESS, self.ACTION_JAM_DECEIVE):

            self.scout.consume_fuel(1.5)

            target_radar = self.radar_system.nearest_radar(
                self.scout.state.position,
                max_range=cfg.JAM_RANGE,
            )

            if target_radar is None:
                # No radar in jamming range: wasted action.
                reward -= 0.5

            elif action == self.ACTION_JAM_SUPPRESS:
                self.radar_system.jam_suppress(
                    target_radar.radar_id,
                    strength=cfg.JAM_SUPPRESSION_STRENGTH,
                    duration=cfg.JAM_SUPPRESSION_DURATION,
                )
                reward += 0.2

            else:
                self.radar_system.jam_deceive(
                    target_radar.radar_id,
                    duration=cfg.JAM_DECEPTION_DURATION,
                )
                reward += 0.2

        else:
            raise ValueError(f"Invalid action: {action}")

        new_x = self.scout.state.position.x
        new_y = self.scout.state.position.y

        info = {
            "scout": {
                "x": new_x,
                "y": new_y,
                "fuel": self.scout.state.fuel,
            },
            "goal": {
                "x": self.strike_point.x,
                "y": self.strike_point.y,
            },
            "action": self.scout_action_name(action),
        }

        return reward, done, info

    def _apply_hunter_action(self, hunter_action: int | None):

        if hunter_action is None:
            self._hunter_heuristic_move()
        else:
            self._hunter_apply_explicit_action(hunter_action)

        hx = self.hunter.state.position.x
        hy = self.hunter.state.position.y

        hunter_reached_target_now = (
            hx == self.strike_point.x and hy == self.strike_point.y
        )
        target_reached_this_step = (
            hunter_reached_target_now and not self.hunter_target_reached
        )
        if target_reached_this_step:
            self.hunter_target_reached = True

        return {
            "x": hx,
            "y": hy,
            "fuel": self.hunter.state.fuel,
            "strike_point": {
                "x": self.strike_point.x,
                "y": self.strike_point.y,
            },
            "target_reached": self.hunter_target_reached,
            "target_reached_this_step": target_reached_this_step,
        }

    def _hunter_heuristic_move(self):
        """
        Greedy pursuit toward the strike point, mirroring the Scout's own
        movement mechanics. This keeps the Hunter active before it becomes
        its own learning agent under MARL (IPPO/MAPPO/IMAHPPO stages).
        """

        if not self.hunter.is_operational():
            return

        hx = self.hunter.state.position.x
        hy = self.hunter.state.position.y

        if hx == self.strike_point.x and hy == self.strike_point.y:
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

            if not self._is_inside_grid(cx, cy):
                continue

            if not self.terrain.is_passable(cx, cy):
                continue
            if(cx, cy) in radar_cells:
                continue

            d = self._dist(cx, cy, self.strike_point.x, self.strike_point.y)

            if d < best_distance:
                best_distance = d
                best = (cx, cy)

        if best is not None:
            self.hunter.move(*best)
            self.hunter.consume_fuel(1.0)

    def _hunter_apply_explicit_action(self, action: int):
        """Apply a movement-only action for future learned Hunter policies."""

        if not self.hunter.is_operational():
            return

        hx = self.hunter.state.position.x
        hy = self.hunter.state.position.y

        new_x, new_y = hx, hy

        if action == self.ACTION_UP:
            new_y -= 1
        elif action == self.ACTION_DOWN:
            new_y += 1
        elif action == self.ACTION_LEFT:
            new_x -= 1
        elif action == self.ACTION_RIGHT:
            new_x += 1
        elif action != self.ACTION_STAY:
            raise ValueError(f"Invalid Hunter action: {action}")

        if self._is_inside_grid(new_x, new_y) and self.terrain.is_passable(
            new_x, new_y
        ):
            self.hunter.move(new_x, new_y)
            self.hunter.consume_fuel(1.0)

    # ==================================================
    # HELPERS
    # ==================================================

    def scout_action_name(self, action: int) -> str:

        names = {
            self.ACTION_UP: "UP",
            self.ACTION_DOWN: "DOWN",
            self.ACTION_LEFT: "LEFT",
            self.ACTION_RIGHT: "RIGHT",
            self.ACTION_STAY: "STAY",
            self.ACTION_JAM_SUPPRESS: "JAM_SUPPRESS",
            self.ACTION_JAM_DECEIVE: "JAM_DECEIVE",
        }

        return names.get(action, "UNKNOWN")

    @staticmethod
    def _dist(x0: int, y0: int, x1: int, y1: int) -> float:

        return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5

    def _is_inside_grid(self, x: int, y: int) -> bool:

        return 0 <= x < self.width and 0 <= y < self.height

    def _get_state(self):

        return {
            "scout": {
                "aircraft_id": self.scout.state.aircraft_id,
                "x": self.scout.state.position.x,
                "y": self.scout.state.position.y,
                "fuel": self.scout.state.fuel,
            },
            "hunter": {
                "aircraft_id": self.hunter.state.aircraft_id,
                "x": self.hunter.state.position.x,
                "y": self.hunter.state.position.y,
                "fuel": self.hunter.state.fuel,
            },
            "weather": self._weather_dict(),
        }

    # ==================================================
    # DASHBOARD LOGGING
    # ==================================================

    def _log_step(self, info: dict):

        terrain_grid = None

        if self.current_step == 1:
            terrain_grid = self.terrain.grid.tolist()

        entry = {
            "step": self.current_step,
            "scout": info["scout"],
            "hunter": info["hunter"],
            "goal": info["goal"],
            "radar_status": info["radar_status"],
            "weather": info["weather"],
            "reward": info["reward"],
            "action": info["action"],
            "hunter_action": info["hunter_action"],
            "lethal_hit": info["lethal_hit"],
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
            entry["terrain_grid"] = terrain_grid
            entry["radars"] = [
                {"radar_id": r.radar_id, "x": r.position.x, "y": r.position.y}
                for r in self.radar_system.radars
            ]
            entry["strike_point"] = {
                "x": self.strike_point.x,
                "y": self.strike_point.y,
            }

        self.episode_log.append(entry)

    def export_log(self, path: str | Path):
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(self.episode_log, file, indent=2)
