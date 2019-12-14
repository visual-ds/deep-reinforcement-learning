'''
Inspired by: Moustafa Alzantot
https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
'''

import gym
import numpy as np


def random_policy(env):
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        policy[s] = np.random.randint(0, env.env.nA)
    return policy

def value_iteration(env, gamma=0.99, eps=1e-20, max_iterations=100000):
    V = np.zeros(env.env.nS)
    Q = np.zeros((env.env.nS, env.env.nA))
    for i in range(max_iterations):
        old_V = np.copy(V)
        for s in range(env.env.nS):
            for a in range(env.env.nA):
                Q[s][a] = sum(p * (gamma * old_V[s_] + r) for p, s_, r, _ in env.env.P[s][a])
            V[s] = max(Q[s])
        if np.sum(np.fabs(V - old_V)) < eps:
            print('Value iteration done with {} iterations.'.format(i))
            break
        elif i % 100 == 0:
            evaluate_policy(env, extract_policy(env, Q),
            text='Our policy is performing well in {}% of times, with ' + str(i) + ' iterations.',
            comparison=False)
    return V, Q

def extract_policy(env, Q):
    policy = np.zeros(env.env.nS, dtype=np.int8)
    for s in range(env.env.nS):
        policy[s] = np.argmax(Q[s])
    return policy

def evaluate_policy(env, policy, n=1000, text='Our policy performs well in {}% of times.', comparison=True):
    success = 0
    for _ in range(n):
        success += run_episode(env, policy)
    print(text.format(success/n*100))
    if comparison:
        success = 0
        for _ in range(n):
            success += run_episode(env, random_policy(env))
        print('Just comparing, random policy performs well in {}% of times.'.format(success/n*100))

def run_episode(env, policy, render1=False, render2=False, max_iterations=10000000):
    obs = env.reset()
    for _ in range(max_iterations):
        if render1:
            env.render()
        obs, reward, done, _ = env.step(policy[obs])
        if done:
            if render2:
                env.render()
            return reward
    return 0

if __name__ == '__main__':
    env = gym.make('FrozenLake8x8-v0')
    V, Q = value_iteration(env)
    opt_policy = extract_policy(env, Q)
    evaluate_policy(env, opt_policy)
    print("An example:")
    run_episode(env, opt_policy, render2=True)