'''
Inspired by: Moustafa Alzantot
https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
'''

import gym
import numpy as np
import os
import pickle


def main():
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()

    env = gym.make('MountainCar-v0')
    if args.run:
        with open('Q-learning.pkl', 'rb') as f:
            policy = pickle.load(f)
    else:
        env.seed(0)
        np.random.seed(0)    
        policy = np.zeros((40, 40))
        Q = np.zeros((40, 40, 3))
        Q = q_learning(env, Q)
        policy = extract_policy(env, Q)
        save_policy(policy)
    scores = [run_episode(env, policy) for _ in range(100)]
    print('The average total reward of our policy is {}.'.format(np.mean(scores)))
    for _ in range(5):
        run_episode(env, policy, render=True)

def run_episode(env, policy=None, max_iterations=100000, render=False):
    obs = env.reset()
    total_reward = 0
    
    for _ in range(max_iterations):
        s0, s1 = obs_to_state(env, obs)
        a = policy[s0][s1]
        obs, reward, done, _ = env.step(a)
        total_reward += reward
        if render:
            env.render()
        if done:
            break
        
    env.close()
    return total_reward

def obs_to_state(env, obs):
    s = np.int8(np.floor(40 * (obs - env.env.low) / (env.env.high - env.env.low)))
    return s[0], s[1]

def q_learning(env, Q, max_iterations1=100000, max_iterations2=10000, gamma=1.0):
    total_reward = 0
    for i in range(max_iterations2):
        obs = env.reset()
        alpha = max(0.003, 1.0 * (0.85 ** (i//100)))
        
        for _ in range(max_iterations1):
            s0, s1 = obs_to_state(env, obs)
            a = take_action(env, s0, s1, Q)
            obs, r, done, _ = env.step(a)
            total_reward += r
            s0_, s1_ = obs_to_state(env, obs)
            Q_obs = r + gamma * np.max(Q[s0_][s1_])
            Q[s0][s1][a] = (1 - alpha) * Q[s0][s1][a] + alpha * Q_obs
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

    policy = np.zeros((40, 40), dtype=np.int8)
    for s0 in range(40):
        for s1 in range(40):
            policy[s0][s1] = np.argmax(Q[s0][s1])
    return policy

def evaluate_policy(env, Q):
    policy = extract_policy(env, Q)
    R = 0
    for _ in range(100):
        R += run_episode(env, policy)
    R /= 1000
    print('The average total reward of this policy is {}.'.format(R))

def take_action(env, s0, s1, Q):
    if np.random.uniform(0, 1) < 0.02:
        a = np.random.choice(env.action_space.n)
    else:
        logits = Q[s0][s1]
        logits_exp = np.exp(logits)
        probs = logits_exp / np.sum(logits_exp)
        a = np.random.choice(env.action_space.n, p=probs)
    return a

def save_policy(policy):
    with open('Q-learning.pkl', 'wb') as f:
        pickle.dump(policy, f, pickle.HIGHEST_PROTOCOL)   


if __name__ == '__main__':
    main()
