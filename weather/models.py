from dataclasses import dataclass


@dataclass
class WeatherState:
    temperature: float
    wind_speed: float
    wind_direction: float
    visibility: float
    precipitation: float
    cloud_cover: float