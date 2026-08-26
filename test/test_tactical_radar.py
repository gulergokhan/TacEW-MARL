import unittest
from unittest.mock import patch

from aircraft.models import Position
from radar.models import Radar, RadarState
from radar.radar import RadarSystem
from terrain.models import TerrainType
from terrain.terrain_map import TerrainMap


class TestTacticalRadar(unittest.TestCase):

    def setUp(self):
        self.radar = Radar(
            radar_id="radar_test",
            position=Position(5, 5),
            detection_range=3.0,
        )
        self.system = RadarSystem([self.radar])

    def test_close_consecutive_detections_escalate_to_lethal(self):
        target = Position(5, 6)

        with patch("radar.radar.random.random", return_value=0.0):
            self.system.detect(target)
            detections = self.system.detect(target)

        self.assertEqual(detections[0]["state"], RadarState.LETHAL)

    def test_track_survives_two_consecutive_misses(self):
        target = Position(5, 7)
        track = self.radar.get_track("scout_01")
        track.state = RadarState.TRACK

        with patch("radar.radar.random.random", return_value=1.0):
            self.system.detect(target)
            self.system.detect(target)

        self.assertEqual(track.state, RadarState.TRACK)

        with patch("radar.radar.random.random", return_value=1.0):
            self.system.detect(target)

        self.assertEqual(track.state, RadarState.SAFE)

    def test_scout_and_hunter_tracks_are_independent(self):
        with patch("radar.radar.random.random", return_value=0.0):
            self.system.detect(Position(9, 9), target_id="scout_01")
            self.system.detect(Position(5, 6), target_id="hunter_01")

        self.assertEqual(
            self.radar.state_for("scout_01"),
            RadarState.SAFE,
        )
        self.assertEqual(
            self.radar.state_for("hunter_01"),
            RadarState.LOCK,
        )

    def test_suppression_jam_reduces_signal_to_jam_ratio(self):
        target = Position(5, 7)
        baseline, _, _, _ = self.system.js_ratio(self.radar, target)

        self.system.jam_suppress(self.radar.radar_id, strength=0.75, duration=5)
        jammed, _, _, _ = self.system.js_ratio(self.radar, target)

        self.assertLess(jammed, baseline)

    def test_mountain_blocks_radar_line_of_sight(self):
        terrain = TerrainMap(10, 10)
        terrain.set_terrain(6, 5, TerrainType.MOUNTAIN)

        _, _, visible, attenuation = self.system.js_ratio(
            self.radar,
            Position(7, 5),
            terrain=terrain,
        )

        self.assertFalse(visible)
        self.assertEqual(attenuation, 0.0)


if __name__ == "__main__":
    unittest.main()
