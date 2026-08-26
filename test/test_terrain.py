from terrain.models import TerrainType
from terrain.terrain_map import TerrainMap


def main():
    terrain = TerrainMap(width=10, height=10)

    terrain.set_terrain(2, 3, TerrainType.MOUNTAIN)
    terrain.set_terrain(5, 5, TerrainType.WATER)
    terrain.set_terrain(7, 2, TerrainType.FOREST)

    print("Terrain Map")
    print("----------------")

    print("(2, 3):", terrain.get_terrain(2, 3).name)
    print("(5, 5):", terrain.get_terrain(5, 5).name)
    print("(7, 2):", terrain.get_terrain(7, 2).name)

    print()
    print("Passability")
    print("----------------")

    print("(2, 3):", terrain.is_passable(2, 3))
    print("(5, 5):", terrain.is_passable(5, 5))
    print("(7, 2):", terrain.is_passable(7, 2))


if __name__ == "__main__":
    main()