import unittest

from terrain.models import TerrainType
from terrain.terrain_map import TerrainMap


class TestTerrainMap(unittest.TestCase):

    def setUp(self):
        self.terrain = TerrainMap(
            width=10,
            height=10,
        )

    def test_water_is_passable(self):
        self.terrain.set_terrain(
            5,
            5,
            TerrainType.WATER,
        )

        self.assertTrue(
            self.terrain.is_passable(5, 5)
        )

    def test_mountain_is_passable_without_altitude_model(self):
        self.terrain.set_terrain(
            3,
            2,
            TerrainType.MOUNTAIN,
        )

        self.assertTrue(
            self.terrain.is_passable(3, 2)
        )

    def test_mountain_blocks_line_of_sight(self):
        self.terrain.set_terrain(
            3,
            2,
            TerrainType.MOUNTAIN,
        )

        visible, attenuation = (
            self.terrain.has_line_of_sight(
                2,
                2,
                4,
                2,
            )
        )

        self.assertFalse(visible)
        self.assertEqual(attenuation, 0.0)

    def test_forest_attenuates_line_of_sight(self):
        self.terrain.set_terrain(
            3,
            2,
            TerrainType.FOREST,
        )

        visible, attenuation = (
            self.terrain.has_line_of_sight(
                2,
                2,
                4,
                2,
            )
        )

        self.assertTrue(visible)
        self.assertAlmostEqual(
            attenuation,
            0.8,
        )


if __name__ == "__main__":
    unittest.main()
