from dataclasses import dataclass
from enum import Enum


class AircraftType(Enum):
    SCOUT = "scout"
    HUNTER = "hunter"


@dataclass
class Position:
    x: int
    y: int


@dataclass
class AircraftState:
    aircraft_id: str
    aircraft_type: AircraftType
    position: Position
    fuel: float
    altitude: float
    speed: float
    # Kinematics: heading in degrees, 0 = North, clockwise (matches wind_direction
    # convention used by the weather module). Updated on every move().
    heading: float = 0.0