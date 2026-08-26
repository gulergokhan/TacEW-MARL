from dataclasses import dataclass
from enum import Enum

from aircraft.models import Position


class RadarState(Enum):
    SAFE = "SAFE"
    TRACK = "TRACK"
    LOCK = "LOCK"
    LETHAL = "LETHAL"


@dataclass
class Radar:
    radar_id: str
    position: Position
    detection_range: float = 3.0
    state: RadarState = RadarState.SAFE
    active: bool = True

    def reset(self):
        self.state = RadarState.SAFE
        self.active = True