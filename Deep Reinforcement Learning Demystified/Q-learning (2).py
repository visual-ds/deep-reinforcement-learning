'''
Inspired by: Moustafa Alzantot
https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
'''

import gym
import numpy as np


def run_episode(env, policy=None, max_iterations=100000, render=False):
    s = env.reset()
    total_reward = 0
    
    for _ in range(max_iterations):
        a = policy[s]
        s, reward, done, _ = env.step(a)
        total_reward += reward
        if render:
            env.render()
        if done:
            break

    return total_reward

def obs_to_state(env, obs):
    s = np.int8(np.floor(40 * (obs - env.env.low) / (env.env.high - env.env.low)))
    return s[0], s[1]

def q_learning(env, Q, max_iterations1=100000, max_iterations2=10000, gamma=1.0):
    total_reward = 0
    for i in range(max_iterations2):
        s = env.reset()
        alpha = max(0.003, 1.0 * (0.85 ** (i//100)))
        
        for _ in range(max_iterations1):
            a = take_action(env, s, Q)
            s_, r, done, _ = env.step(a)
            total_reward += r
            Q_obs = r + gamma * np.max(Q[s_])
            Q[s][a] = (1 - alpha) * Q[s][a] + alpha * Q_obs
            s = s_
            if done:
                break

        if i % 100 == 0:
            total_reward /= 100
            print('Iteration {} completed: average total reward = {}.'.format(i, total_reward))
            total_reward = 0

    return Q

def extract_policy(env, Q):
    '''
    Extract policy from Q function.
    '''

    policy = np.zeros(64, dtype=np.int8)
    for s in range(64):
        policy[s] = np.argmax(Q[s])
    return policy

def evaluate_policy(env, Q):
    policy = extract_policy(env, Q)
    R = 0
    for _ in range(100):
        R += run_episode(env, policy)
    R /= 1000
    print('The average total reward of this policy is {}.'.format(R))

def take_action(env, s, Q):
    if np.random.uniform(0, 1) < 0.02:
        a = np.random.choice(env.action_space.n)
    else:
        logits = Q[s]
        logits_exp = np.exp(logits)
        probs = logits_exp / np.sum(logits_exp)
        a = np.random.choice(env.action_space.n, p=probs)
    return a


if __name__ == '__main__':
    env = gym.make('FrozenLake8x8-v0')
    env.seed(0)
    np.random.seed(0)
    Q = np.zeros((64, 4))
    Q = q_learning(env, Q)
    policy = extract_policy(env, Q)
    scores = [run_episode(env, policy) for _ in range(100)]
    print('The average total reward of our policy is {}.'.format(np.mean(scores)))
