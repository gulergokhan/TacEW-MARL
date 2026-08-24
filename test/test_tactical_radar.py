from environment.tactical_env import TacticalEnv


def main():

    env = TacticalEnv(
        width=10,
        height=10,
        latitude=39.9334,
        longitude=32.8597,
    )

    env.reset()

    print("Tactical Radar Test")
    print("======================")

    print(
        f"Initial Scout Position: "
        f"({env.scout.state.position.x}, "
        f"{env.scout.state.position.y})"
    )

    print()
    print("Moving Scout toward Radar 01")
    print("----------------------")

    # (1,1) -> (2,1)
    env.step(env.ACTION_RIGHT)

    # (2,1) -> (3,1)
    env.step(env.ACTION_RIGHT)

    # (3,1) -> (4,1)
    env.step(env.ACTION_RIGHT)

    # (4,1) -> (5,1)
    env.step(env.ACTION_RIGHT)

    # (5,1) -> (5,2)
    env.step(env.ACTION_DOWN)

    # (5,2) -> (5,3)
    env.step(env.ACTION_DOWN)

    # (5,3) -> (5,4)
    env.step(env.ACTION_DOWN)

    # (5,4) -> (5,5)
    observation, reward, done, info = env.step(
        env.ACTION_DOWN
    )

    print(
        f"Scout Position: "
        f"({env.scout.state.position.x}, "
        f"{env.scout.state.position.y})"
    )

    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Radar Detections: {info['radar_detections']}")

    print()
    print("Radar States")
    print("----------------------")

    for radar in env.radar_system.radars:

        print(
            radar.radar_id,
            "->",
            radar.state.value
        )


if __name__ == "__main__":
    main()