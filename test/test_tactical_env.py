from environment.tactical_env import TacticalEnv


def main():
    env = TacticalEnv(
        width=10,
        height=10,
        latitude=39.9334,
        longitude=32.8597,
    )

    observation = env.reset()

    print("Initial Observation")
    print("----------------------")
    print(observation)
    print("Shape:", observation.shape)
    print("Data type:", observation.dtype)

    print()

    print("Scout moves RIGHT")
    print("----------------------")

    next_observation, reward, done, info = env.step(
        TacticalEnv.ACTION_RIGHT
    )

    print("Next Observation:", next_observation)
    print("Shape:", next_observation.shape)
    print("Reward:", reward)
    print("Done:", done)


if __name__ == "__main__":
    main()