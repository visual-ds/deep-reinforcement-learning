from collections import deque
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import gym
import numpy as np
import os
import pickle
import random


class NeuralNetwork():
    def __init__(self, obs_size, action_size):
        self.obs_size = obs_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.model_of_network()

    def model_of_network(self):
        self.model = Sequential()
        self.model.add(Dense(units=24, input_shape=(self.obs_size,), activation='relu'))
        self.model.add(Dense(units=48, activation='relu'))
        self.model.add(Dense(units=self.action_size, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))


class Agent():
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.env.seed(0)
        np.random.seed(0)    
        self.gamma = 1.0
        self.n_episodes = 500
        self.max_iterations = 200
        self.obs_size = self.env.env.observation_space.shape[0]
        self.action_size = self.env.env.action_space.n
        self.epsilon = 1.0
        self.epsilon_decay = 0.05
        self.epsilon_min = 0.01
        self.epsilon_max_iterations = 1000
        self.replay_memory_capacity = 20000
        self.n_epochs = 100
        self.minibatch_size = 32

    def read_policy(self, policy_path):
        '''
        If there's already a policy, read it.
        '''

        with open(policy_path, 'rb') as f:
            self.policy = pickle.load(f)

    def train(self):
        '''
        Train deep Q-learning agent.
        '''

        self.deep_q_learning()

    def deep_q_learning(self):
        '''
        Deep Q-learning.
        It learns the Q function without knowing the transition probabilities, through deep neural networks.
        '''

        self.replay_memory = deque(maxlen=self.replay_memory_capacity)
        self.neural_network = NeuralNetwork(self.obs_size, self.action_size)

        for episode in range(self.n_episodes):
            self.train_episode(episode)

        self.env.close()

    def train_episode(self, episode):
        obs = self.env.reset()
        state = obs#self.normalize_obs(obs)
        total_reward = 0
        for i in range(self.max_iterations):
            action = self.take_action(state)
            obs, reward, done, _ = self.env.step(action)
            state_ = obs#self.normalize_obs(obs)
            self.replay_memory.append([state, action, reward, state_, done])
            self.train_from_replay()
            # experience = self.sample_experience()
            # y = self.set_target(experience, done)
            # self.update_network(y, experience)
            state = state_
            total_reward += reward
            if done:
                break
        result = 'FAIL'
        if i < 199:
            result = 'SUCESS'
        print('Ep. {}: {}. Reward = {} and epsilon = {}.'.format(episode, result, total_reward, self.epsilon), end='\n\n')
        self.epsilon -= self.epsilon_decay
        self.epsilon = np.max([self.epsilon, self.epsilon_min])

    def train_from_replay(self):
        if len(self.replay_memory) < self.minibatch_size:
            return
        minibatch = random.sample(self.replay_memory, self.minibatch_size)
        states = []
        targets = []
        for sample in minibatch:
            state, action, reward, state_, done = sample
            states.append(state)
            target = self.compute_values(state)
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma*np.max(self.compute_values(state_))
            targets.append(target)

        states = np.array(states)
        targets = np.array(targets)

        self.neural_network.model.fit(
            x=states,
            y=targets,
            verbose=0
        )

            
    def normalize_obs(self, obs):
        '''
        Normalize observation.
        Raw observation -> observation between 0 and 1
        '''

        state = (obs - self.env.env.low) / (self.env.env.high - self.env.env.low)
        return state
            
    def take_action(self, state):
        '''
        Take action based in epsilon greedy algorithm.
        With small probabily, take a random action;
        otherwise, take the action from the neural network.
        '''
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.env.action_space.n)
        else:
            output = self.compute_values(state)
            action = np.argmax(output)
        return action

    def sample_experience(self):
        """Sample experience from replay memory."""
        return self.replay_memory[np.random.choice(len(self.replay_memory))]

    def set_target(self, experience, done):
        """Set target for neural network update."""
        state, action, reward, state_ = experience
        if done:
            return reward
        max_Q = np.max(self.compute_values(state_))
        return reward + self.gamma*max_Q

    def update_network(self, y, experience):
        """Update neural network from replay_memory."""
        state, action, reward, state_ = experience
        input_ = state[None, :]
        target = self.neural_network.model.predict(input_)
        target[0, action] = y
        self.neural_network.model.fit(input_, target, verbose=0, epochs=self.n_epochs)

    def compute_values(self, state):
        '''
        Compute values of a state.
        '''

        logits = self.neural_network.model.predict(state[None, :])
        return logits[0, :]

    def save_network(self, network_path):
        '''
        Save the network in order to run it faster.
        '''

        with open(network_path, 'wb') as f:
            pickle.dump(self.neural_network.model, f, pickle.HIGHEST_PROTOCOL)

    def read_network(self, network_path):
        '''
        Read the network in order to run it faster.
        '''
        self.neural_network.model
        with open(network_path, 'rb') as f:
            self.neural_network.model = pickle.load(f)

    def sample(self):
        '''
        It shows a sample of our policy.
        '''

        for _ in range(5):
            _ = self.run_episode(render=True)

    def run_episode(self, render=False):
        '''
        It runs one episode for our policy.
        '''

        obs = self.env.reset()
        total_reward = 0
        for _ in range(self.max_iterations):
            state = self.normalize_obs(obs)
            action = self.which_action(state)
            obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            if render:
                self.env.render()
            if done:
                break

        self.env.close()
        return total_reward

    def which_action(self, state):
        return np.argmax(self.compute_values(state))

    def test(self):
        scores = [self.run_episode() for _ in range(10)]
        print('Average total reward: {}.'.format(np.mean(scores)))


def main():
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()

    agent = Agent('MountainCar-v0')

    if args.run:
        agent.read_network('deep_q_learning.pkl')
        agent.sample()
    else:
        agent.train()
        agent.save_network('deep_q_learning.pkl')
    agent.test()


if __name__ == '__main__':
    main()
