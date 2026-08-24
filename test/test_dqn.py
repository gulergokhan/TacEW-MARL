from environment.tactical_env import TacticalEnv
from dqn.agent import DQNAgent


def main():

    env = TacticalEnv(
        width=10,
        height=10,
        latitude=39.9334,
        longitude=32.8597,
    )

    observation = env.reset()

    agent = DQNAgent(
        observation_size=57,
        action_size=5,
    )

    print("DQN Test")
    print("----------------------")
    print("Observation size:", len(observation))
    print("Action size:", agent.action_size)
    print("Device:", agent.device)
    print("Initial epsilon:", agent.epsilon)

    # Fill replay buffer
    for step in range(100):

        action = agent.select_action(
            observation
        )

        next_observation, reward, done, info = env.step(
            action
        )

        agent.remember(
            observation,
            action,
            reward,
            next_observation,
            done,
        )

        loss = agent.learn()

        observation = next_observation

        if done:
            observation = env.reset()

    print()
    print("After Training Steps")
    print("----------------------")
    print("Replay buffer:", len(agent.memory))
    print("Learn steps:", agent.learn_step_count)
    print("Last loss:", loss)
    print("Epsilon:", agent.epsilon)


if __name__ == "__main__":
    main()