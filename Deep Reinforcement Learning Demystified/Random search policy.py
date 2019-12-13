'''
Inspired by: Moustafa Alzantot
https://medium.com/@m.alzantot/deep-reinforcement-learning-demystified-episode-0-2198c05a6124
'''
import gym
import numpy as np


def take_action(policy, obs):
    if np.dot(policy, np.append(obs, 1)) > 0:
        return 0
    else:
        return 1

def random_policy():
    policy = np.random.uniform(-1, 1, 5)
    return policy

def run_episode(env, policy, render=False):
    obs = env.reset()
    totalReward = 0
    for _ in range(1000):
        if render:
            env.render()
        obs, reward, done, _ = env.step(take_action(policy, obs))
        totalReward += reward
        if done:
            break
    return totalReward


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    policies = [random_policy() for _ in range(500)]
    totalRewards = [run_episode(env, policy) for policy in policies]
    for _ in range(5):
        run_episode(env, policies[np.argmax(totalRewards)], render=True)