'''
Inspired by: Moustafa Alzantot
https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
'''

import gym
import numpy as np


def random_policy(env):
    policy = np.zeros(env.env.nS, dtype=np.int8)
    for s in range(env.env.nS):
        policy[s] = np.random.randint(0, env.env.nA)
    return policy

def calculate_V_policy(env, policy, gamma, eps=1e-10, max_iterations=10000):
    '''
    It calculates the value function (under our policy) iteratively.
    It could be done solving the linear equations proposed in the inspired article (on the top).
    '''

    V_policy = np.zeros(env.env.nS)

    for _ in range(max_iterations):
        old_V_policy = np.copy(V_policy)
        for s in range(env.env.nS):
            V_policy[s] = sum(p * (gamma * old_V_policy[s_] + r) for p, s_, r, _ in env.env.P[s][policy[s]])
        if np.sum(np.fabs(old_V_policy - V_policy)) < eps:
            break

    return V_policy

def extract_policy(env, V_policy, gamma):
    '''
    Extract the policy from the value function (under the policy).
    '''

    aux = np.zeros((env.env.nS, env.env.nA))
    policy_ = np.zeros(env.env.nS, dtype=np.int8)

    for s in range(env.env.nS):
        for a in range(env.env.nA):
            aux[s][a] = sum(p * (gamma * V_policy[s_] + r) for p, s_, r, _ in env.env.P[s][a])
        policy_[s] = np.argmax(aux[s])

    return policy_

def evaluate_policy(env, policy, n=1000, text='Our policy performs well in {}% of times.', comparison=True):
    '''
    Evaluate the policy computing the percent of times it wins.
    '''

    success = 0
    for _ in range(n):
        success += run_episode(env, policy)
    print(text.format(success/n*100))

    if comparison:
        success = 0
        random_policy_ = random_policy(env)
        for _ in range(n):
            success += run_episode(env, random_policy_)
        print('Just comparing, random policy performs well in {}% of times.'.format(success/n*100))

def run_episode(env, policy, render1=False, render2=False, max_iterations=10000000):
    '''
    Run one episode with the specified policy.
    '''

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

def policy_iteration(env, gamma=0.99, max_iterations=10000):
    '''
    Policy iteration algorithm.
    It finds the optimal policy and value function iteratively.
    '''

    policy_ = random_policy(env)

    for i in range(max_iterations):
        policy = np.copy(policy_)
        V_policy = calculate_V_policy(env, policy, gamma)
        policy_ = extract_policy(env, V_policy, gamma)
        if np.all(policy == policy_):
            break
        elif i % 1 == 0:
            # Evaluate our policy during iteration.
            evaluate_policy(env, policy_, text='Our policy is performing well in {}% of times, after iteration ' + str(i) + '.',
            comparison=False)

    return policy_


if __name__ == '__main__':
    env = gym.make('FrozenLake8x8-v0')
    policy_ = policy_iteration(env)
    evaluate_policy(env, policy_)
    print("An example:")
    run_episode(env, policy_, render2=True)