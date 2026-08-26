from dataclasses import dataclass, field
from enum import Enum

from aircraft.models import Position


class RadarState(Enum):
    SAFE = "SAFE"
    TRACK = "TRACK"
    LOCK = "LOCK"
    LETHAL = "LETHAL"


@dataclass
class RadarTrack:
    state: RadarState = RadarState.SAFE
    lock_streak: int = 0
    miss_streak: int = 0


@dataclass
class Radar:
    radar_id: str
    position: Position
    detection_range: float = 3.0
    state: RadarState = RadarState.SAFE
    active: bool = True

    # Transmit power proxy used in the J/S ratio calculation.
    base_power: float = 1.0

    # Consecutive steps spent in LOCK before escalating to LETHAL.
    lethal_lock_threshold: int = 2

    # Tracking memory prevents a single probabilistic miss from immediately
    # dropping an otherwise valid track.
    max_missed_steps: int = 2

    # Each aircraft has an independent tracking state.
    tracks: dict[str, RadarTrack] = field(default_factory=dict)

    # Jamming state (set by suppression/deception actions, decays over time).
    suppression_jam: float = 0.0
    suppression_timer: int = 0
    deception_jam: bool = False
    deception_timer: int = 0

    def reset(self):
        self.state = RadarState.SAFE
        self.active = True
        self.tracks.clear()
        self.suppression_jam = 0.0
        self.suppression_timer = 0
        self.deception_jam = False
        self.deception_timer = 0

    def tick_jam(self):
        if self.suppression_timer > 0:
            self.suppression_timer -= 1
            if self.suppression_timer == 0:
                self.suppression_jam = 0.0

        if self.deception_timer > 0:
            self.deception_timer -= 1
            if self.deception_timer == 0:
                self.deception_jam = False

    def get_track(self, target_id: str) -> RadarTrack:
        if target_id not in self.tracks:
            self.tracks[target_id] = RadarTrack()
        return self.tracks[target_id]

    def state_for(self, target_id: str) -> RadarState:
        return self.get_track(target_id).state

    def sync_aggregate_state(self):
        priority = {
            RadarState.SAFE: 0,
            RadarState.TRACK: 1,
            RadarState.LOCK: 2,
            RadarState.LETHAL: 3,
        }
        self.state = max(
            (track.state for track in self.tracks.values()),
            key=priority.get,
            default=RadarState.SAFE,
        )
