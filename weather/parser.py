from weather.models import WeatherState


class WeatherParser:

    @staticmethod
    def parse(data: dict) -> WeatherState:
        current = data["current"]

        return WeatherState(
            temperature=current["temperature_2m"],
            wind_speed=current["wind_speed_10m"],
            wind_direction=current["wind_direction_10m"],
            visibility=current["visibility"],
            precipitation=current["precipitation"],
            cloud_cover=current["cloud_cover"],
        )