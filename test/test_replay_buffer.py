from agent.replay_buffer import ReplayBuffer


def main():

    buffer = ReplayBuffer(capacity=100)

    buffer.add(
        [0.0, 0.0],
        3,
        -1,
        [0.0, 0.111],
        False
    )

    buffer.add(
        [0.0, 0.111],
        1,
        -1,
        [0.111, 0.111],
        False
    )

    print("Buffer size:", len(buffer))

    states, actions, rewards, next_states, dones = buffer.sample(2)

    print("States:", states)
    print("Actions:", actions)
    print("Rewards:", rewards)
    print("Next states:", next_states)
    print("Dones:", dones)


if __name__ == "__main__":
    main()