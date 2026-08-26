from weather.client import WeatherClient
from weather.parser import WeatherParser


def main():
    client = WeatherClient()

    data = client.get_weather(
    latitude=39.9334,
    longitude=32.8597
)

    weather = WeatherParser.parse(data)

    print("Weather State")
    print("----------------")
    print(f"Temperature   : {weather.temperature} °C")
    print(f"Wind Speed    : {weather.wind_speed} km/h")
    print(f"Wind Direction: {weather.wind_direction}°")
    print(f"Visibility    : {weather.visibility} m")
    print(f"Precipitation : {weather.precipitation} mm")
    print(f"Cloud Cover   : {weather.cloud_cover}%")


if __name__ == "__main__":
    main()