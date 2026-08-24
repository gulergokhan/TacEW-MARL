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
    def distance(position_a: Position, position_b: Position) -> float:
        dx = position_a.x - position_b.x
        dy = position_a.y - position_b.y

        return math.sqrt(dx ** 2 + dy ** 2)

    def detect(self, aircraft_position: Position):
        detections = []

        for radar in self.radars:

            if not radar.active:
                continue

            distance = self.distance(
                radar.position,
                aircraft_position,
            )

            if distance <= radar.detection_range:
                radar.state = RadarState.TRACK

                detections.append({
                    "radar_id": radar.radar_id,
                    "distance": distance,
                    "state": radar.state,
                })

            else:
                radar.state = RadarState.SAFE

        return detections