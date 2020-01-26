from keras.layers import Dense
from keras.models import Sequential
import gym
import numpy as np
import os
import pickle


class NeuralNetwork():
    def __init__(self, obs_size, action_size):
        self.obs_size = obs_size
        self.action_size = action_size
        self.model_of_network()

    def model_of_network(self):
        self.model = Sequential()
        self.model.add(Dense(units=int(np.ceil(1.2*self.obs_size)), input_shape=(self.obs_size,), activation='relu'))
        self.model.add(Dense(units=self.action_size))
        self.model.compile(loss='mse', optimizer='adam')


class Agent():
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.env.seed(0)
        np.random.seed(0)    
        self.gamma = 1.0

        self.obs_size = self.env.env.observation_space.shape[0]
        self.action_size = self.env.env.action_space.n
        self.epsilon = 1.0
        self.iterations = 0
        self.replay_memory_capacity = 1000000

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

    def deep_q_learning(self, n_episodes=10000, max_iterations=100000):
        '''
        Deep Q-learning.
        It learns the Q function without knowing the transition probabilities, through deep neural networks.
        '''

        self.replay_memory = []
        self.neural_network = NeuralNetwork(self.obs_size, self.action_size)

        for episode in range(n_episodes):
            obs = self.env.reset()
            state = self.normalize_obs(obs)
            for _ in range(max_iterations):
                if self.iterations%10000 == 0:
                    self.test()
                action = self.take_action(state)
                obs, reward, done, _ = self.env.step(action)
                state_ = self.normalize_obs(obs)
                self.append_replay_memory([state, action, reward, state_])
                experience = self.sample_experience()
                y = self.set_target(experience, done)
                self.update_network(y, experience)
                self.iterations += 1
                state = state_
                if done:
                    break

        self.env.close()
            
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

        # Epsilon-greedy algorithm
        if self.iterations <= 1000000:
            self.epsilon = -9e-7 * self.iterations + 1
        else:
            self.epsilon = 0.1

        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.env.action_space.n)
        else:
            output = self.compute_values(state)
            action = np.argmax(output)
        return action

    def append_replay_memory(self, experience):
        """Append replay memory with experience."""
        if len(self.replay_memory) >= self.replay_memory_capacity:
            self.replay_memory.pop(0)
        self.replay_memory.append(experience)

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
        self.neural_network.model.fit(input_, target, verbose=0)

        # input_batch, target_batch = self.get_input_and_target(batch)
        # self.neural_network.model.fit(input_batch, target_batch, verbose=0, epochs=300)

    def get_input_and_target(self, batch):
        '''
        Get input and target for training neural network.
        '''

        input_batch = np.empty((0, self.obs_size+self.action_size))
        target_batch = np.empty((0, 1))

        for experience in batch:
            state, action, reward, state_ = experience
            action_array = np.zeros(self.action_size)
            action_array[action] = 1
            input = np.concatenate((state, action_array))
            input_batch = np.append(input_batch, input[None, :], axis=0)

            target = np.array([[reward + self.gamma*self.value(state_)]])
            target_batch = np.append(target_batch, target, axis=0)
        
        return input_batch, target_batch

    def value(self, state):
        '''
        Get value of a state. It's the maximum of the values of the state.
        '''

        output = self.compute_values(state)
        value = np.max(output)
        return value

    def compute_values(self, state):
        '''
        Compute values of a state.
        '''

        logits = self.neural_network.model.predict(state[None, :])
        return logits[0, :]

        # output = []
        # for action in range(self.action_size):
        #     action_array = np.zeros(self.action_size)
        #     action_array[action] = 1
        #     input = np.concatenate((state, action_array))
        #     logits = self.neural_network.model.predict(input[None, :])
        #     output.append(logits[0, 0])
        # return output

    def save_network(self, network_path):
        '''
        Save the network in order to run it faster.
        '''

        with open(network_path, 'wb') as f:
            pickle.dump(self.neural_network.model, f, pickle.HIGHEST_PROTOCOL)

    def save_policy(self, policy_path):
        '''
        Save the policy in a file, allowing to run it later.
        '''

        with open(policy_path, 'wb') as f:
            pickle.dump(self.policy, f, pickle.HIGHEST_PROTOCOL)

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

    def run_episode(self, render=False, max_iterations=100000):
        '''
        It runs one episode for our policy.
        '''

        obs = self.env.reset()
        total_reward = 0
        for _ in range(max_iterations):
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
        print('The average total reward of our policy is {}.'.format(np.mean(scores)))


def main():
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()

    agent = Agent('MountainCar-v0')

    if args.run:
        agent.read_network('Q-learning.pkl')
        agent.sample()
    else:
        agent.train()
        agent.save_network('Q-learning.pkl')
    agent.test()


if __name__ == '__main__':
    main()
