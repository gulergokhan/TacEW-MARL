from aircraft.aircraft import Aircraft
from aircraft.models import AircraftType

from weather.client import WeatherClient
from weather.parser import WeatherParser

from terrain.terrain_map import TerrainMap

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
        # It will not be requested again
        # during every environment reset.
        self.weather = self._load_weather()

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

    # ==================================================
    # WEATHER
    # ==================================================

    def _load_weather(self):
        """
        Load weather data from the Weather API.
        This method is called only once when
        TacticalEnv is created.
        """

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
        """
        Reset the tactical environment.

        Aircraft positions and fuel are reset.

        Weather data is NOT requested again.
        """

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

        state = self._get_state()

        observation = self.observation_encoder.encode(
            state,
            self.terrain,
        )

        return observation

    # ==================================================
    # STEP
    # ==================================================

    def step(self, action: int):
        """
        Execute one action for the scout aircraft.
        """

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
        # Grid boundary
        # -----------------------------

        if not self._is_inside_grid(
            new_x,
            new_y,
        ):
            reward = -1.0

        # -----------------------------
        # Terrain passability
        # -----------------------------

        elif not self.terrain.is_passable(
            new_x,
            new_y,
        ):
            reward = -1.0

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
        # Fuel check
        # -----------------------------

        if self.scout.state.fuel <= 0:
            done = True

        # -----------------------------
        # New state
        # -----------------------------

        state = self._get_state()

        observation = self.observation_encoder.encode(
            state,
            self.terrain,
        )

        return (
            observation,
            reward,
            done,
            {},
        )

    # ==================================================
    # GRID CHECK
    # ==================================================

    def _is_inside_grid(
        self,
        x: int,
        y: int,
    ) -> bool:
        """
        Check whether the given position
        is inside the environment grid.
        """

        return (
            0 <= x < self.width
            and 0 <= y < self.height
        )

    # ==================================================
    # STATE
    # ==================================================

    def _get_state(self):
        """
        Build the current tactical state.
        """

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