# Inspired by 'Playing Atari with Deep Reinforcement Learning' (https://arxiv.org/abs/1312.5602)

from collections import deque
from datetime import datetime
# from gym import wrappers
import argparse
import cProfile
import csv
import gym
import numpy as np
import os
import pickle
import random
import tensorflow as tf
import time


class NeuralNetwork():
    def __init__(self, input_shape, action_size, learning_rate=0.00025, summary=False):
        self.input_shape = input_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.summary = summary
        self.model = self.model_of_network()

    def model_of_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                input_shape=self.input_shape,
                filters=16,
                kernel_size=8,
                strides=(4, 4),
                activation='relu'
            ),
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=4,
                strides=(2, 2),
                activation='relu'
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=256,
                activation='relu'
            ),
            tf.keras.layers.Dense(
                units=self.action_size
            )
        ])
        if self.summary:
            model.summary()
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate))
        return model

class ReplayMemory():
    def __init__(self, maxlen, minibatch_size, n_channels):
        self.replay_memory = deque(maxlen=maxlen)
        self.minibatch_size = minibatch_size
        self.n_channels = n_channels

    def append(self, experience):
        """Apend experience to replay memory."""
        self.replay_memory.append(experience)

    def last_4_states(self, state):
        """Concatenate last four states."""
        state = state[..., np.newaxis]  # Actual state
        for i in range(3):  # Last three states in replay memory
            state = np.concatenate((state, self.replay_memory[-i-1][0][..., np.newaxis]), axis=2)        
        return state / 255

    def minibacth(self):
        """Sample a minibatch from the memory replay."""
        # Choose self.minibatch_size indices from replay memory
        i_minibatch = random.sample(range(3, len(self.replay_memory) - 1), self.minibatch_size)
        # Get these samples
        minibatch = [self.replay_memory[i].copy() for i in i_minibatch]
        # Fill up the states and states_
        my_minibatch = minibatch.copy()
        for i, _ in enumerate(minibatch):
            state = self.full_state(i_minibatch[i])
            my_minibatch[i][0] = state
            state_ = self.full_state(i_minibatch[i]+1)
            my_minibatch[i].insert(3, state_)
        minibatch = my_minibatch
        return minibatch

    def full_state(self, i):
        """Get the full state of the i sample."""
        state = self.replay_memory[i][0][..., np.newaxis]
        for j in range(3):
            state = np.concatenate((state, self.replay_memory[i-j-1][0][..., np.newaxis]), axis=2)
        return state / 255

class DeepQAgent():
    def __init__(
        self,
        record,
        env_name='BreakoutDeterministic-v4',
        gamma=0.99,
        max_frames=10000000,
        max_iterations=1000000,
        epsilon_decay_until=10000000,
        epsilon_min=0.1,
        replay_memory_capacity=1000000,
        start_replay=50000,
        minibatch_size=32,
        nn_input_shape=(105, 80, 4),
    ):
        self.env = gym.make(env_name)
        # if record:
        #     self.env = wrappers.Monitor(self.env, os.path.join(os.getcwd(), 'videos', str(time.time())))
        self.set_seeds(int(time.time()))
        self.gamma = gamma
        self.max_frames = max_frames
        self.max_iterations = max_iterations
        self.obs_shape = self.env.observation_space.shape
        self.action_size = self.env.env.action_space.n
        self.epsilon = 1.0
        self.epsilon_decay_until = epsilon_decay_until
        self.epsilon_min = epsilon_min
        self.replay_memory_capacity = replay_memory_capacity
        # self.replay_memory = deque(maxlen=self.replay_memory_capacity)
        self.start_replay = start_replay
        self.minibatch_size = minibatch_size
        self.nn_input_shape = nn_input_shape
        self.neural_network = NeuralNetwork(self.nn_input_shape, self.action_size, summary=True)
        self.i_frames = 0
        self.replay_memory = ReplayMemory(
            maxlen=self.replay_memory_capacity,
            minibatch_size=self.minibatch_size,
            n_channels = self.nn_input_shape[2]
        )
        self.mean_reward = deque(maxlen=100)
        self.rewards = []

    def set_seeds(self, seed):
        """Set random seeds using current time."""
        self.env.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def train(self):
        """Train deep Q-learning agent."""
        self.start_time = time.time()
        self.fill_replay_memory()
        self.deep_q_learning()

    def fill_replay_memory(self):
        """Fill replay memory on start."""

        print('\nFilling replay memory...', end='')
        while self.i_frames < self.start_replay:
            obs = self.env.reset()
            self.state = self.preprocess(obs)
            for _ in range(self.max_iterations):
                action = self.env.action_space.sample()
                obs, reward, done, _ = self.env.step(action)
                reward = self.transform_reward(reward)
                self.replay_memory.append([self.state, action, reward, done])
                self.state = self.preprocess(obs)
                self.i_frames += 1
                if not self.i_frames < self.start_replay:
                    break
                elif done:
                    break
        print(' Done.')

    def preprocess(self, obs):
        """Preprocess observation."""
        img = self.resize(obs)
        img = self.to_gray_scale(img)
        # img = self.cut(gray)
        # state = self.create_state(img[..., np.newaxis])
        return img.astype(np.uint8)
    
    def resize(self, img):
        img = img[::2, ::2]
        return img

    def to_gray_scale(self, img):
        return np.mean(img, axis=2)     

    def transform_reward(self, reward):
        return np.sign(reward)

    def deep_q_learning(self):
        """
        Deep Q-learning algorithm.
        It learns the Q function without knowing the transition probabilities, through deep neural networks.
        """
        # Copy network
        self.target_network = NeuralNetwork(self.nn_input_shape, self.action_size)
        self.target_network.model.set_weights(self.neural_network.model.get_weights())

        episode = 0  # Flag
        while self.i_frames < self.max_frames:
            if self.train_episode(episode):
                break
            # if (episode + 1) % 50 == 0:
            #     self.sample(1)
            episode += 1

        self.env.close()

    def train_episode(self, episode):
        """Train one episode of deep Q-learning."""
        # Flags
        self.time_flag = time.time()
        self.frames_flag = self.i_frames
        max_frames = False

        total_reward = 0  # Cumulative

        obs = self.env.reset()
        self.state = self.preprocess(obs)

        for i in range(self.max_iterations):
            action = self.take_action()
            obs, reward, done, _ = self.env.step(action)
            reward = self.transform_reward(reward)
            self.replay_memory.append([self.state, action, reward, done])
            self.i_frames += 1
            total_reward += reward

            self.state = self.preprocess(obs)
            self.train_from_replay()
            if not self.i_frames < self.max_frames:  # Max frames
                max_frames = True
                break
            elif done:
                break
        self.rewards.append(total_reward)
        self.report(i, episode, total_reward)
        self.sync_networks()
        if (episode + 1) % 100 == 0:
            now = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
            self.save_network('models/' + now + '-breakout-{}.h5'.format(episode))
        return max_frames

    def take_action(self):
        """
        Take action based in epsilon-greedy algorithm.
        With small probabily, take a random action;
        otherwise, take the action from the neural network.
        """
        self.update_epsilon()
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_size)
        else:
            last_4_states = self.replay_memory.last_4_states(self.state)
        #     state = self.replay_memory[-1][0][..., np.newaxis]
        #     for i in range(1, 4):
        #         state = np.concatenate((state, self.replay_memory[-1-i][0][..., np.newaxis]), axis=2)
            output = self.compute_values(last_4_states)
            action = np.argmax(output)
        return action

    def update_epsilon(self):
        """Update epsilon for epsilon-greedy algorithm."""
        epsilon = 1 + (self.i_frames * self.epsilon_min - self.i_frames) / self.epsilon_decay_until
        self.epsilon = np.max([self.epsilon_min, epsilon])

    def compute_values(self, last_4_states):
        """Compute values of a state."""
        logits = self.neural_network.model.predict(last_4_states[np.newaxis, ...])
        return logits[0]

    # def train_from_replay(self):
    #     minibatch = self.replay_memory.minibatch()
    #     states = []
    #     actions = []
    #     rewards = []
    #     states_ = []
    #     dones = []
    #     for sample in minibatch:
    #         state, action, reward, state_, done = sample
    #         states.append(state)
    #         actions.append(action)
    #         rewards.append(reward)
    #         states_.append(state_)
    #         dones.append(done)
    #     targets = self.neural_network.model.predict(states)
    #     targets_ = self.target_network.model.predict(states_)
    #     for i, target in enumerate(targets):
    #         if dones[i]:
    #             targets[i][actions[i]] = rewards[i]
    #         else:
    #             targets[i][actions[i]] = rewards[i] + self.gamma * np.max(targets_[i])
    #     self.neural_network.model.fit(states, targets)

    def train_from_replay(self):
        """Train neural network from samples of replay memory."""
        minibatch = self.replay_memory.minibacth()
        states, states_ = self.extract_states(minibatch)
        targets = self.calculate_targets(minibatch, states, states_)
        self.fit(states, targets)

    def extract_states(self, minibatch):
        states = []
        states_ = []
        for sample in minibatch:
            state, _, _, state_, _ = sample
            states.append(state)
            states_.append(state_)
        states = np.array(states)
        states_ = np.array(states_)
        return states, states_

    def calculate_targets(self, minibatch, states, states_):
        targets, targets_ = self.target_predict(states, states_)
        for i, sample in enumerate(minibatch):
            _, action, reward, _, done = sample
            if done:
                targets[i][action] = reward
            else:
                targets[i][action] = reward + self.gamma * np.max(targets_[i])
        return targets

    def target_predict(self, states, states_):
        targets = self.neural_network.model.predict(states)
        targets_ = self.target_network.model.predict(states_)
        return targets, targets_

    def fit(self, states, targets):
        self.neural_network.model.fit(
            x=states,
            y=targets,
            verbose=0
        )

    def report(self, i, episode, total_reward):
        """Show status on console."""

        self.mean_reward.append(total_reward)

        print()
        print('ep    reward    mean_rew    epsilon    time     accum_time    frames    acumm_frames')

        ep = episode
        print('{}'.format(ep) + ' '*(len('ep')+4-len(str(ep))), end='')

        reward = int(total_reward)
        print('{}'.format(reward) + ' '*(len('reward')+4-len(str(reward))), end='')

        mean_rwd = np.mean(self.mean_reward)
        mean_rwd = '{:.1f}'.format(mean_rwd)
        print(mean_rwd + ' '*(len('mean_rew')+4-len(mean_rwd)), end='')

        epsilon = self.epsilon
        epsilon = '{:.3f}'.format(epsilon)
        print(epsilon + ' '*(len('epsilon')+4-len(epsilon)), end='')

        time_ = time.time() - self.time_flag
        time_ = '{:.2f}'.format(time_)
        print(time_ + ' '*(len('time')+5-len(time_)), end='')

        accum_time = time.time() - self.start_time
        accum_time = '{:.2f}'.format(accum_time)
        print(accum_time + ' '*(len('accum_time')+4-len(accum_time)), end='')

        frames = self.i_frames - self.frames_flag
        print('{}'.format(frames) + ' '*(len('frames')+4-len(str(frames))), end='')

        accum_frames = self.i_frames
        print('{}'.format(accum_frames) + ' '*(len('accum_frames')+4-len(str(accum_frames))))

    def sync_networks(self):
        """Sync original and target neural networks."""
        self.target_network.model.set_weights(self.neural_network.model.get_weights())

    def save_network(self, network_path):
        """Save the network in order to run it faster."""
        os.makedirs(os.path.dirname(network_path), exist_ok=True)
        self.neural_network.model.save(network_path)
        print('\nNeural network saved.')

    def load_network(self, network_path):
        """Load the network in order to run it faster."""
        self.neural_network.model = tf.keras.models.load_model(network_path)
        print('\nNeural network loaded.', end='\n\n')

    def sample(self, n=5):
        """Sample the network."""
        print('\nSampling network:\n')
        for _ in range(n):
            self.run_episode(render=True)

    def run_episode(self, render=False):
        """Run one episode for our network."""
        total_reward = 0        
        obs = self.env.reset()
        state = self.preprocess(obs)[..., np.newaxis]
        for i in range(self.max_iterations):
            if i < 3:
                action = self.env.action_space.sample()
                obs, reward, done, _ = self.env.step(action)
                state = np.concatenate((self.preprocess(obs)[..., np.newaxis], state), axis=2)
            else:
                action = self.which_action(state / 255)
                obs, reward, done, _ = self.env.step(action)
                state = np.concatenate((self.preprocess(obs)[..., np.newaxis], state[..., :3]), axis=2)
            total_reward += reward
            if render:
                self.env.render()
            if done:
                break
            time.sleep(0.01)
        self.env.close()
        print('Reward: ', total_reward)

    def which_action(self, state):
        """Select which is the best action based on the network."""
        if np.random.rand() < self.epsilon_min/10:
            return self.env.action_space.sample()
        logits = self.neural_network.model.predict(state[np.newaxis, ...])[0]
        return np.argmax(logits)

    def save_rewards(self, file_path):
        """Save the list of total rewards."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(self.rewards)
        print('\nRewards saved.')

def gpu_setup():
    """Config GPU for TensorFlow."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print()
                print(len(gpus), "physical GPUs,", len(logical_gpus), "logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def main():
    parser = argparse.ArgumentParser(description='Train or run Breakout deep Q-learning algorithm.')
    parser.add_argument('--run', metavar='MODEL_PATH', help='run saved model')
    parser.add_argument('--gpu', action='store_true', help='limit gpu growth')
    parser.add_argument('--profile', action='store_true', help='profile training')
    # parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    # agent = DeepQAgent(args.record)
    if args.gpu:
        gpu_setup()
    agent = DeepQAgent(False)
    if args.run:
        agent.load_network(args.run)
        agent.sample()
    else:
        if args.profile:
            cProfile.runctx('agent.train()', {'agent': agent}, {}, filename='stats/temp-breakout')
            now = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
            os.rename('stats/temp-breakout', 'stats/' + now + '-breakout')
            print('\nProfile saved.')
        else:
            agent.train()        
            now = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        agent.save_rewards('rewards/' + now + '-breakout.csv')
        agent.save_network('models/' + now + '-breakout.h5')

if __name__ == '__main__':
    main()
