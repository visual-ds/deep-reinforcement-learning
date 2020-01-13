'''
Inspired by: Moustafa Alzantot
https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
'''

import gym
import numpy as np
import os
import pickle


class Agent():

    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.env.seed(0)
        np.random.seed(0)    
        self.policy = np.zeros((40, 40))
        self.Q = np.zeros((40, 40, 3))

    def read_policy(self, policy_path):
        with open(policy_path, 'rb') as f:
            self.policy = pickle.load(f)

    def train(self):
        self.q_learning()
        self.extract_policy()

    def q_learning(self, max_iterations1=100000, max_iterations2=10000, gamma=1.0):
        '''
        Q-learning algorithm.
        It learns the Q function without knowing the transition probabilities.
        '''

        total_reward = 0
        for i in range(max_iterations2):
            obs = self.env.reset()
            alpha = max(0.003, 1.0 * (0.85 ** (i//100)))
            
            for _ in range(max_iterations1):
                s0, s1 = self.obs_to_state(obs)
                a = self.take_action(s0, s1)
                obs, r, done, _ = self.env.step(a)
                total_reward += r
                s0_, s1_ = self.obs_to_state(obs)
                self.update_Q(r, gamma, s0_, s1_, s0, s1, a, alpha)
                if done:
                    break

            if i % 100 == 0:
                total_reward /= 100
                print('Iteration {} completed: average total reward = {}.'.format(i, total_reward))
                total_reward = 0

        self.env.close()
            
    def update_Q(self, r, gamma, s0_, s1_, s0, s1, a, alpha):
        Q_obs = r + gamma * np.max(self.Q[s0_][s1_])
        self.Q[s0][s1][a] = (1 - alpha) * self.Q[s0][s1][a] + alpha * Q_obs

    def extract_policy(self):
        '''
        Extract policy from Q function.
        '''

        self.policy = np.zeros((40, 40), dtype=np.int8)
        for s0 in range(40):
            for s1 in range(40):
                self.policy[s0][s1] = np.argmax(self.Q[s0][s1])

    def save_policy(self, policy_path):
        '''
        Save the policy in a file, allowing to run it later.
        '''

        with open(policy_path, 'wb') as f:
            pickle.dump(self.policy, f, pickle.HIGHEST_PROTOCOL)

    def test(self):
        scores = [self.run_episode() for _ in range(100)]
        print('The average total reward of our policy is {}.'.format(np.mean(scores)))

    def run_episode(self, max_iterations=100000, render=False):
        '''
        Run one episode for the policy.
        '''

        obs = self.env.reset()
        total_reward = 0
        
        for _ in range(max_iterations):
            s0, s1 = self.obs_to_state(obs)
            a = self.policy[s0][s1]
            obs, reward, done, _ = self.env.step(a)
            total_reward += reward
            if render:
                self.env.render()
            if done:
                break
            
        self.env.close()
        return total_reward

    def sample(self):
        for _ in range(5):
            self.run_episode(render=True)

    def obs_to_state(self, obs):
        '''
        Because observations are continuous, we've discretized it.
        So, we convert it to integer numbers.
        '''

        s = np.int8(np.floor(40 * (obs - self.env.env.low) / (self.env.env.high - self.env.env.low)))
        return s[0], s[1]

    def take_action(self, s0, s1):
        '''
        Take one action during Q-learning, based on our current Q function.
        '''

        if np.random.uniform(0, 1) < 0.02:
            # With small prob., take a random action
            a = np.random.choice(self.env.action_space.n)
        else:
            logits = self.Q[s0][s1]
            logits_exp = np.exp(logits)
            probs = logits_exp / np.sum(logits_exp)
            a = np.random.choice(self.env.action_space.n, p=probs)

        return a


def main():
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    # Just run the saved file
    args = parser.parse_args()

    agent = Agent('MountainCar-v0')

    if args.run:
        agent.read_policy('Q-learning.pkl')
    else:
        agent.train()
        agent.save_policy('Q-learning.pkl')

    agent.test()
    agent.sample()


if __name__ == '__main__':
    main()
