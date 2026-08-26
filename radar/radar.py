import math

from aircraft.models import Position
from radar.models import Radar, RadarState


class RadarSystem:

    def __init__(self, radars=None):
        self.radars = radars if radars is not None else []

    def add_radar(self, radar: Radar):
        self.radars.append(radar)

    def reset(self):
        for radar in self.radars:
            radar.reset()

    @staticmethod
    def distance(
        position_a: Position,
        position_b: Position
    ) -> float:

        dx = position_a.x - position_b.x
        dy = position_a.y - position_b.y

        return math.sqrt(
            dx ** 2 + dy ** 2
        )

    def detect(
        self,
        aircraft_position: Position
    ):

        detections = []

        for radar in self.radars:

            if not radar.active:
                continue

            distance = self.distance(
                radar.position,
                aircraft_position
            )

            if distance <= radar.detection_range:

                if distance <= radar.detection_range * 0.5:
                    radar.state = RadarState.LOCK

                else:
                    radar.state = RadarState.TRACK

                detections.append({
                    "radar_id": radar.radar_id,
                    "distance": distance,
                    "state": radar.state,
                })

            else:

                radar.state = RadarState.SAFE

        return detections

    def get_status(
        self,
        aircraft_position: Position
    ):

        status = []

        for radar in self.radars:

            distance = self.distance(
                radar.position,
                aircraft_position
            )

            status.append({
                "radar_id": radar.radar_id,
                "distance": distance,
                "state": radar.state.value,
                "active": radar.active,
                "detection_range": radar.detection_range,
            })

        return status

    def get_radar(
        self,
        radar_id: str
    ):

        for radar in self.radars:

            if radar.radar_id == radar_id:
                return radar

        return None

    def suppress_radar(
        self,
        radar_id: str
    ) -> bool:

        radar = self.get_radar(
            radar_id
        )

        if radar is None:
            return False

        radar.active = False
        radar.state = RadarState.SAFE

        return True