# deep-reinforcement-learning

Scientific Initiation (2019 - today) in Deep Reinforcement Learning at Getulio Vargas Foundation, supervised by [Dr. Jorge Poco](https://github.com/jpocom).

If you wanna know the current state of my studies, see my [presentation](https://github.com/lucasresck/deep-reinforcement-learning/blob/master/presentations/partial_presentation.pdf).

In this repository, you'll find implementations of:
- *CartPole-v0* random search policy
- *FrozenLake8x8-v0* value iteration
- *FrozenLake8x8-v0* policy iteration
- *MountainCar-v0* Q-learning
- ***MountainCar-v0* deep Q-learning**

![alt text](https://raw.githubusercontent.com/lucasresck/deep-reinforcement-learning/master/images/mountaincar-v0.gif)

## Installation

Clone this repository:

```bash
git clone https://github.com/lucasresck/deep-reinforcement-learning.git
```

## Usage

All Python files can be run in this way:

```bash
python mountaincar_deep_q_learning.py
```

Some files have special arguments:

### *MountainCar-v0* deep Q-learning

For training and saving the model:

```bash
python mountaincar_deep_q_learning.py
```

For running the saved model:

```bash
python mountaincar_deep_q_learning.py --run
```

For saving a video from the trained model:

```bash
python mountaincar_deep_q_learning.py --run --record
```

### *MountainCar-v0* Q-learning

For training and saving the model:

```bash
python mountaincar_q_learning.py
```

For running the saved model:

```bash
python mountaincar_q_learning.py --run
``` 
