# reinforced-race
A model car learns driving along a track using reinforcement learning

## Unity virtual environment

This is a virtual environment the car can learn to drive in.
Built with Unity it enables you to test your new parameters, tracks and algorithms before you take your car out to reality.

Open the [Unity](https://unity3d.com/) project in `rl-race-virtual-env` and run the game, then start the learning algorithm.

## The Deep-Q-Learning reinforcement learning algorithm

The reinforcement learning algorithm is inspired from [Mnhi et al. 2013](https://arxiv.org/abs/1312.5602).
After starting the virtual environment run

```
python3 rl-race-learning/qlearner.py
```

The module will connect to the virtual environment using a TCP connection.

See `python3 qlearner.py --help` for more options.

## The real model car

*Coming soon*
