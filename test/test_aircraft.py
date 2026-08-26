from aircraft.aircraft import Aircraft
from aircraft.models import AircraftType


def main():
    scout = Aircraft(
        aircraft_id="scout_01",
        aircraft_type=AircraftType.SCOUT,
        start_x=1,
        start_y=2,
    )

    hunter = Aircraft(
        aircraft_id="hunter_01",
        aircraft_type=AircraftType.HUNTER,
        start_x=8,
        start_y=8,
    )

    print("Aircraft Test")
    print("----------------")

    print(
        scout.state.aircraft_id,
        scout.state.aircraft_type.value,
        scout.state.position,
        scout.state.fuel,
    )

    print(
        hunter.state.aircraft_id,
        hunter.state.aircraft_type.value,
        hunter.state.position,
        hunter.state.fuel,
    )

    scout.move(2, 3)
    scout.consume_fuel(10)

    print()
    print("After Scout Update")
    print("----------------")

    print("Position:", scout.state.position)
    print("Fuel:", scout.state.fuel)
    print("Operational:", scout.is_operational())


if __name__ == "__main__":
    main()