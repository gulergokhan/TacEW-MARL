from environment.gridworld import GridWorld
from agent.dqn_agent import DQNAgent


def main():

    env = GridWorld()

    state = env.reset()

    agent = DQNAgent(
        state_size=len(state),
        action_size=4,
        batch_size=4
    )

    print("Initial State:", state)

    for step in range(5):

        action = agent.select_action(state)

        next_state, reward, done, info = env.step(action)

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        loss = agent.learn()

        print(
            f"Step={step + 1} | "
            f"Action={action} | "
            f"Reward={reward} | "
            f"Loss={loss}"
        )

        state = next_state

        if done:
            break


if __name__ == "__main__":
    main() 