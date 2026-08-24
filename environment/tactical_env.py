from aircraft.aircraft import Aircraft
from aircraft.models import AircraftType, Position

from weather.client import WeatherClient
from weather.parser import WeatherParser

from terrain.terrain_map import TerrainMap

from radar.models import Radar
from radar.radar import RadarSystem

from environment.observation import ObservationEncoder


class TacticalEnv:

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_STAY = 4

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        latitude: float = 39.9334,
        longitude: float = 32.8597,
    ):
        self.width = width
        self.height = height

        self.latitude = latitude
        self.longitude = longitude

        # Maximum number of steps per episode
        self.max_steps = 100

        # Goal position
        self.goal_position = Position(
            x=8,
            y=8,
        )

        # -----------------------------
        # Terrain
        # -----------------------------

        self.terrain = TerrainMap(
            width=width,
            height=height,
        )

        # -----------------------------
        # Weather
        # -----------------------------

        self.weather_client = WeatherClient()

        # Weather is loaded only once.
        self.weather = self._load_weather()

        # -----------------------------
        # Radar
        # -----------------------------

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

        # -----------------------------
        # Observation Encoder
        # -----------------------------

        self.observation_encoder = ObservationEncoder(
            width=width,
            height=height,
        )

        # -----------------------------
        # Aircraft
        # -----------------------------

        self.scout = None
        self.hunter = None

        # Step counter
        self.current_step = 0

    # ==================================================
    # WEATHER
    # ==================================================

    def _load_weather(self):
        weather_data = self.weather_client.get_weather(
            latitude=self.latitude,
            longitude=self.longitude,
        )

        return WeatherParser.parse(
            weather_data
        )

    # ==================================================
    # RESET
    # ==================================================

    def reset(self):

        self.current_step = 0

        self.scout = Aircraft(
            aircraft_id="scout_01",
            aircraft_type=AircraftType.SCOUT,
            start_x=1,
            start_y=1,
        )

        self.hunter = Aircraft(
            aircraft_id="hunter_01",
            aircraft_type=AircraftType.HUNTER,
            start_x=8,
            start_y=8,
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

    def step(self, action: int):

        self.current_step += 1

        old_x = self.scout.state.position.x
        old_y = self.scout.state.position.y

        new_x = old_x
        new_y = old_y

        # -----------------------------
        # Action
        # -----------------------------

        if action == self.ACTION_UP:
            new_y -= 1

        elif action == self.ACTION_DOWN:
            new_y += 1

        elif action == self.ACTION_LEFT:
            new_x -= 1

        elif action == self.ACTION_RIGHT:
            new_x += 1

        elif action == self.ACTION_STAY:
            pass

        else:
            raise ValueError(
                f"Invalid action: {action}"
            )

        # -----------------------------
        # Default reward
        # -----------------------------

        reward = -0.1
        done = False

        # -----------------------------
        # STAY penalty
        # -----------------------------

        if action == self.ACTION_STAY:
            reward = -0.2

        # -----------------------------
        # Grid boundary
        # -----------------------------

        if not self._is_inside_grid(
            new_x,
            new_y,
        ):
            reward -= 1.0

        # -----------------------------
        # Terrain
        # -----------------------------

        elif not self.terrain.is_passable(
            new_x,
            new_y,
        ):
            reward -= 1.0

        # -----------------------------
        # Valid movement
        # -----------------------------

        else:

            self.scout.move(
                new_x,
                new_y,
            )

            self.scout.consume_fuel(
                1.0
            )

        # -----------------------------
        # Radar Detection
        # -----------------------------

        detections = self.radar_system.detect(
            self.scout.state.position
        )

        if detections:
            reward -= 5.0

        # -----------------------------
        # Distance to Goal
        # -----------------------------

        old_distance = self._distance_to_goal(
            old_x,
            old_y,
        )

        new_distance = self._distance_to_goal(
            self.scout.state.position.x,
            self.scout.state.position.y,
        )

        # Reward getting closer
        if new_distance < old_distance:
            reward += 0.2

        # Penalty moving away
        elif new_distance > old_distance:
            reward -= 0.1

        # -----------------------------
        # Goal
        # -----------------------------

        if (
            self.scout.state.position.x
            == self.goal_position.x
            and
            self.scout.state.position.y
            == self.goal_position.y
        ):
            reward += 20.0
            done = True

        # -----------------------------
        # Fuel
        # -----------------------------

        if self.scout.state.fuel <= 0:
            done = True

        # -----------------------------
        # Maximum steps
        # -----------------------------

        if self.current_step >= self.max_steps:
            done = True

        # -----------------------------
        # Observation
        # -----------------------------

        state = self._get_state()

        observation = self.observation_encoder.encode(
            state,
            self.terrain,
            self.radar_system,
        )

        # -----------------------------
        # Info
        # -----------------------------

        info = {
            "radar_detections": detections,
            "goal_reached": (
                self.scout.state.position.x
                == self.goal_position.x
                and
                self.scout.state.position.y
                == self.goal_position.y
            ),
            "step": self.current_step,
        }

        return (
            observation,
            reward,
            done,
            info,
        )

    # ==================================================
    # DISTANCE
    # ==================================================

    def _distance_to_goal(
        self,
        x: int,
        y: int,
    ) -> float:

        dx = x - self.goal_position.x
        dy = y - self.goal_position.y

        return (dx ** 2 + dy ** 2) ** 0.5

    # ==================================================
    # GRID CHECK
    # ==================================================

    def _is_inside_grid(
        self,
        x: int,
        y: int,
    ) -> bool:

        return (
            0 <= x < self.width
            and
            0 <= y < self.height
        )

    # ==================================================
    # STATE
    # ==================================================

    def _get_state(self):

        return {
            "scout": {
                "x": self.scout.state.position.x,
                "y": self.scout.state.position.y,
                "fuel": self.scout.state.fuel,
            },

            "hunter": {
                "x": self.hunter.state.position.x,
                "y": self.hunter.state.position.y,
                "fuel": self.hunter.state.fuel,
            },

            "weather": {
                "temperature": self.weather.temperature,
                "wind_speed": self.weather.wind_speed,
                "wind_direction": self.weather.wind_direction,
                "visibility": self.weather.visibility,
                "precipitation": self.weather.precipitation,
                "cloud_cover": self.weather.cloud_cover,
            },
        }