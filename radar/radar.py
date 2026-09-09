import math
import random

from aircraft.models import Position
from radar.models import Radar, RadarState, RadarTrack


class RadarSystem:

    def __init__(self, radars=None):
        self.radars = radars if radars is not None else []

    def add_radar(self, radar: Radar):
        self.radars.append(radar)

    def reset(self):
        for radar in self.radars:
            radar.reset()

    def advance_time(self):
        """Advance shared jammer timers once per environment step."""
        for radar in self.radars:
            radar.tick_jam()

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

    # ==================================================
    # WEATHER EFFECTS
    # ==================================================

    @staticmethod
    def _weather_range_factor(weather: dict) -> float:
        """
        Precipitation and cloud cover attenuate the radar's effective
        propagation range. Returns a multiplier in [0.35, 1.0].
        """

        if not weather:
            return 1.0

        precipitation = weather.get("precipitation", 0.0)
        cloud_cover = weather.get("cloud_cover", 0.0)

        precip_factor = max(0.0, 1.0 - (precipitation / 50.0) * 0.4)
        cloud_factor = max(0.0, 1.0 - (cloud_cover / 100.0) * 0.15)

        return max(0.35, precip_factor * cloud_factor)

    @staticmethod
    def _weather_jam_factor(weather: dict) -> float:
        """
        Heavy precipitation also attenuates the jammer's signal on its
        way to the radar, mildly reducing jam effectiveness.
        """

        if not weather:
            return 1.0

        precipitation = weather.get("precipitation", 0.0)

        return max(0.5, 1.0 - (precipitation / 50.0) * 0.25)

    # ==================================================
    # J/S RATIO
    # ==================================================

    def js_ratio(
        self,
        radar: Radar,
        aircraft_position: Position,
        weather: dict = None,
        terrain=None,
    ):
        """
        Returns (js_ratio, distance, visible, attenuation).
        js_ratio > ~1.0 means the signal dominates the jam noise floor.
        """

        distance = max(
            self.distance(radar.position, aircraft_position),
            0.1,
        )

        visible, attenuation = True, 1.0

        if terrain is not None:
            visible, attenuation = terrain.has_line_of_sight(
                radar.position.x, radar.position.y,
                aircraft_position.x, aircraft_position.y,
            )

        if not visible:
            return 0.0, distance, False, 0.0

        range_factor = self._weather_range_factor(weather)

        # Radar-equation-style proxy: signal falls off with distance^2,
        # scaled by terrain attenuation and weather propagation loss.
        signal = (
            radar.base_power
            * attenuation
            * range_factor
            / (distance ** 2)
        )

        jam_strength = radar.suppression_jam * self._weather_jam_factor(weather)
        noise_floor = 0.05
        jam = noise_floor + jam_strength * 4.0

        js = signal / jam

        return js, distance, True, attenuation

    # ==================================================
    # DETECTION
    # ==================================================

    @staticmethod
    def _register_miss(radar: Radar, track: RadarTrack):
        """Keep a track briefly and drop it only after repeated misses."""

        track.miss_streak += 1
        track.lock_streak = max(0, track.lock_streak - 1)

        if track.miss_streak > radar.max_missed_steps:
            track.state = RadarState.SAFE
            track.lock_streak = 0
        elif track.state in (RadarState.LOCK, RadarState.LETHAL):
            track.state = RadarState.TRACK

        radar.sync_aggregate_state()

    def detect(
        self,
        aircraft_position: Position,
        weather: dict = None,
        terrain=None,
        target_id: str = "scout_01",
    ):

        detections = []

        for radar in self.radars:
            if not radar.active:
                continue

            track = radar.get_track(target_id)

            js, distance, visible, _ = self.js_ratio(
                radar, aircraft_position, weather, terrain,
            )

            if not visible or distance > radar.detection_range:
                track.state = RadarState.SAFE
                track.lock_streak = 0
                track.miss_streak = 0
                radar.sync_aggregate_state()
                continue

            # Probabilistic detection from the J/S ratio (logistic curve
            # centered at js == 1.0, i.e. signal == jam noise floor).
            p_detect = 1.0 / (1.0 + math.exp(-2.0 * (js - 1.0)))

            # Deception jamming corrupts the picture: the radar sometimes
            # reports a false SAFE reading even with a real signal present.
            if radar.deception_jam and random.random() < 0.5:
                self._register_miss(radar, track)
                continue

            if random.random() >= p_detect:
                self._register_miss(radar, track)
                continue

            track.miss_streak = 0

            if distance <= radar.detection_range * 0.5 and js >= 1.5:
                track.state = RadarState.LOCK
                track.lock_streak += 1
            else:
                track.state = RadarState.TRACK
                track.lock_streak = 0

            if (
                track.state == RadarState.LOCK
                and track.lock_streak >= radar.lethal_lock_threshold
            ):
                track.state = RadarState.LETHAL

            radar.sync_aggregate_state()

            detections.append({
                "radar_id": radar.radar_id,
                "target_id": target_id,
                "distance": distance,
                "js_ratio": js,
                "state": track.state,
            })

        return detections

    def get_status(
        self,
        aircraft_position: Position,
        weather: dict = None,
        terrain=None,
        target_id: str = "scout_01",
    ):

        status = []

        for radar in self.radars:

            track = radar.get_track(target_id)

            js, distance, visible, _ = self.js_ratio(
                radar, aircraft_position, weather, terrain,
            )

            status.append({
                "radar_id": radar.radar_id,
                "target_id": target_id,
                "distance": distance,
                "js_ratio": js,
                "visible": visible,
                "state": track.state.value,
                "active": radar.active,
                "detection_range": radar.detection_range,
                "suppression_jam": radar.suppression_jam,
                "suppression_timer": radar.suppression_timer,
                "deception_jam": radar.deception_jam,
                "deception_timer": radar.deception_timer,
                "miss_streak": track.miss_streak,
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
        radar.tracks.clear()

        return True

    # ==================================================
    # JAMMING ACTIONS
    # ==================================================

    def jam_suppress(
        self,
        radar_id: str,
        strength: float = 0.75,
        duration: int = 5,
    ) -> bool:
        """Reduces the radar's effective signal-to-jam ratio for `duration` steps."""

        radar = self.get_radar(radar_id)

        if radar is None:
            return False

        radar.suppression_jam = max(radar.suppression_jam, strength)
        radar.suppression_timer = duration

        return True

    def jam_deceive(
        self,
        radar_id: str,
        duration: int = 5,
    ) -> bool:
        """Makes the radar intermittently report false SAFE readings."""

        radar = self.get_radar(radar_id)

        if radar is None:
            return False

        radar.deception_jam = True
        radar.deception_timer = duration

        return True

    def nearest_radar(
        self,
        position: Position,
        max_range: float = None,
        exclude_suppressed: bool = False,
        exclude_deceived: bool = False,
    ):
        """Return the nearest active radar matching the jam filters."""

        nearest = None
        nearest_distance = float("inf")

        for radar in self.radars:

            if not radar.active:
                continue

            if (
                exclude_suppressed
                and radar.suppression_timer > 0
            ):
                continue

            if (
                exclude_deceived
                and radar.deception_timer > 0
            ):
                continue

            distance = self.distance(radar.position, position)

            if distance < nearest_distance:
                nearest = radar
                nearest_distance = distance

        if nearest is None:
            return None

        if max_range is not None and nearest_distance > max_range:
            return None

        return nearest
