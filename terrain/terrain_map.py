import numpy as np

from terrain.models import TerrainType


class TerrainMap:

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        self.grid = np.full(
            (height, width),
            TerrainType.PLAIN,
            dtype=np.int8
        )

    def set_terrain(
        self,
        x: int,
        y: int,
        terrain_type: TerrainType
    ):
        self.grid[y, x] = terrain_type

    def get_terrain(self, x: int, y: int) -> TerrainType:
        return TerrainType(self.grid[y, x])

    def is_passable(self, x: int, y: int) -> bool:
        return self.get_terrain(x, y) != TerrainType.WATER

    # ==================================================
    # LINE OF SIGHT / RADAR SHADOWING
    # ==================================================

    def has_line_of_sight(self, x0: int, y0: int, x1: int, y1: int):
        """
        Checks whether a radar at (x0, y0) can 'see' a target at (x1, y1).

        MOUNTAIN cells fully block line of sight (radar shadow).
        URBAN and FOREST cells attenuate the signal without blocking it.
        Returns (visible: bool, attenuation: float in (0, 1]).
        """

        attenuation = 1.0

        for (x, y) in self._bresenham_line(x0, y0, x1, y1)[1:-1]:

            if not (0 <= x < self.width and 0 <= y < self.height):
                continue

            terrain_type = self.get_terrain(x, y)

            if terrain_type == TerrainType.MOUNTAIN:
                return False, 0.0

            elif terrain_type == TerrainType.URBAN:
                attenuation *= 0.6

            elif terrain_type == TerrainType.FOREST:
                attenuation *= 0.8

        return True, attenuation

    @staticmethod
    def _bresenham_line(x0: int, y0: int, x1: int, y1: int):

        points = []

        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)

        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        err = dx + dy

        x, y = x0, y0

        while True:

            points.append((x, y))

            if x == x1 and y == y1:
                break

            e2 = 2 * err

            if e2 >= dy:
                err += dy
                x += sx

            if e2 <= dx:
                err += dx
                y += sy

        return points
