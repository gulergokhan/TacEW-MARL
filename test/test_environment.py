import random
from environment.gridworld import GridWorld


ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
}


def main():
    env = GridWorld()

    state = env.reset()

    print("Initial State:", state)
    print("Initial Position:", env.scout.position)
    print("-" * 70)

    done = False
    total_reward = 0

    while not done:

        # 0, 1, 2, 3 arasından rastgele action seç
        action = random.randint(0, 3)

        next_state, reward, done, info = env.step(action)

        total_reward += reward

        print(
            f"Step={info['step']:3} | "
            f"Action={ACTION_NAMES[action]:5} | "
            f"Position={info['scout_position']} | "
            f"Reward={reward:4} | "
            f"Detected={info['detected']}"
        )

    print("-" * 70)
    print("Episode finished.")
    print("Total reward:", total_reward)
    print("Final position:", env.scout.position)


if __name__ == "__main__":
    main()