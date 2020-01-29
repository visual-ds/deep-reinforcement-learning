# Deep-RL-Scientific-Initiation

Scientific Initiation in Deep Reinforcement Learning at Getulio Vargas Foundation.

In this repo, you'll find:
- *CartPole-v0* random search policy
- *FrozenLake8x8-v0* value iteration
- *FrozenLake8x8-v0* policy iteration
- *MountainCar-v0* Q-learning
- ***MountainCar-v0* deep Q-learning**

![alt text](https://raw.githubusercontent.com/lucasresck/Deep-RL-Scientific-Initiation/master/videos/1580174020.5228155/openaigym.video.0.17462.video000000.gif)

## Installation

Clone this repository:

```bash
git clone https://github.com/lucasresck/Deep-RL-Scientific-Initiation.git
```

## Usage

All Python files can be run in this way:

```bash
python deep_q_learning.py
```

Some files have special arguments:

### *MountainCar-v0* deep Q-learning

For training and saving the model:

```bash
python deep_q_learning.py
```

For running the saved model:

```bash
python deep_q_learning.py --run
```

For saving a video from the trained model:

```bash
python deep_q_learning.py --run --record
```

### *MountainCar-v0* Q-learning

For training and saving the model:

```bash
python q_learning.py
```

For running the saved model:

```bash
python q_learning.py --run
``` 
