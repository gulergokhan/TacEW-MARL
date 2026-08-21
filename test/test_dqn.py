from environment.gridworld import GridWorld
from agent.dqn_agent import DQNAgent


def main():

    env = GridWorld()

    state = env.reset()

    agent = DQNAgent(
        state_size=len(state),
        action_size=4
    )

    print("Initial State:", state)

    action = agent.select_action(state)

    print("Selected Action:", action)

    next_state, reward, done, info = env.step(action)

    print("Next State:", next_state)
    print("Reward:", reward)
    print("Done:", done)


if __name__ == "__main__":
    main()