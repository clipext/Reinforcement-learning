import gym.spaces
import matplotlib.pyplot as plt
import numpy as np
from qlearning_template import QLearningAgent

def play_and_train(env, agent, t_max=10 ** 4):
    """ This function should
    - run a full game (for t_max steps), actions given by agent
    - train agent whenever possible
    - return total reward
    """
    total_reward = 0.0
    state = env.reset()
    for t in range(t_max):
        action = agent.get_action(state)
        new_state, reward, is_done, _ = env.step(action)
        agent.update(state, action, new_state, reward)
        total_reward += reward
        state = new_state
        if is_done:
            break

    return total_reward


def update_epsilon(model, alpha=0.99, t=0.1):
    if model.epsilon > t:
        model.epsilon *= alpha


if __name__ == '__main__':
    max_iterations = 1000
    visualize = True
    # Create environment 'Taxi-v2'
    env = gym.make('Taxi-v2')
    env.reset()
    env.render()

    # Compute number of states for this environment
    n_states = env.env.nS
    # Compute number of actions for this environment
    n_actions = env.env.nA

    print('States number = %i, Actions number = %i' % (n_states, n_actions))

    # create q learning agent with
    # alpha=0.5
    # get_legal_actions = lambda s: range(n_actions)
    # epsilon=0.1
    # discount=0.99

    agent = QLearningAgent(alpha=0.5, epsilon=0.1, discount=0.99, get_legal_actions=lambda s: range(n_actions))

    plt.figure(figsize=[10, 4])
    sarsa_rewards = []
    evsarsa_rewards = []
    rewards = []

    # Training loop
    for i in range(max_iterations):
        rewards.append(play_and_train(env, agent))

        # Play & train game
        # Update rewards
        # rewards

        # Decay agent epsilon
        # agent.epsilon = ?
        update_epsilon(agent)

        if i % 100 == 0:
            print('Iteration {}, Average reward {:.2f}, Epsilon {:.3f}'.format(i, np.mean(rewards), agent.epsilon))


        if visualize:
            plt.subplot(1, 2, 1)
            plt.plot(rewards, color='r')
            plt.xlabel('Iterations')
            plt.ylabel('Total Reward')

            plt.subplot(1, 2, 2)
            plt.hist(rewards, bins=20, range=[-700, +20], color='blue', label='Rewards distribution')
            plt.xlabel('Reward')
            plt.ylabel('p(Reward)')
            plt.draw()
            plt.pause(0.05)
            plt.cla()
