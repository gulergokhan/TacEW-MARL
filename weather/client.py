import requests


class WeatherClient:

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def get_weather(self, latitude: float, longitude: float) -> dict:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,"
                "wind_speed_10m,"
                "wind_direction_10m,"
                "visibility,"
                "precipitation,"
                "cloud_cover"
            ),
        }

        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        return response.json()