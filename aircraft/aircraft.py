from aircraft.models import AircraftState, AircraftType, Position


class Aircraft:

    def __init__(
        self,
        aircraft_id: str,
        aircraft_type: AircraftType,
        start_x: int,
        start_y: int,
        fuel: float = 100.0,
        altitude: float = 1000.0,
        speed: float = 10.0,
    ):
        self.state = AircraftState(
            aircraft_id=aircraft_id,
            aircraft_type=aircraft_type,
            position=Position(start_x, start_y),
            fuel=fuel,
            altitude=altitude,
            speed=speed,
        )

    def move(self, x: int, y: int):
        self.state.position.x = x
        self.state.position.y = y

    def consume_fuel(self, amount: float):
        self.state.fuel = max(0.0, self.state.fuel - amount)

    def is_operational(self) -> bool:
        return self.state.fuel > 0.0