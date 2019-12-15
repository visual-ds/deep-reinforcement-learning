import gym
import numpy as np


def run_episode(env, policy=None, max_iterations=100000, render=False):
    _ = env.reset()
    if render:
        env.render()
    
    for _ in range(max_iterations):
        _, _, done, _ = env.step(env.action_space.sample())
        if render:
            env.render()
        if done:
            break

def q_learning(env, Q, max_iterations=100000):
    for _ in range(max_iterations):
        break
    pass


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    Q = np.random.randint(0, 3, (40, 40), np.int8)
    Q = q_learning(env, Q)
    print(Q)