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