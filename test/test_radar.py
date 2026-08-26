from aircraft.models import Position

from radar.models import Radar
from radar.radar import RadarSystem


def main():

    radar = Radar(
        radar_id="radar_01",
        position=Position(x=5, y=5),
        detection_range=3.0,
    )

    radar_system = RadarSystem(
        radars=[radar]
    )

    print("Radar Test")
    print("======================")

    print(f"Radar: {radar.radar_id}")
    print(
        f"Position: "
        f"({radar.position.x}, {radar.position.y})"
    )
    print(
        f"Detection Range: "
        f"{radar.detection_range}"
    )
    print(
        f"Initial State: "
        f"{radar.state.value}"
    )

    print()
    print("Aircraft at (1, 1)")
    print("----------------------")

    detections = radar_system.detect(
        Position(x=1, y=1)
    )

    print("Detections:", detections)
    print("Radar State:", radar.state.value)

    print()
    print("Aircraft at (4, 5)")
    print("----------------------")

    detections = radar_system.detect(
        Position(x=4, y=5)
    )

    print("Detections:", detections)
    print("Radar State:", radar.state.value)


if __name__ == "__main__":
    main()