#!/usr/bin/python3

import argparse

from racelearning import models
from racelearning.environment import LeftRightAction, Action, EnvironmentInterface
from racelearning.qlearner import AnnealingRAPolicy, QLearner
from racelearning.qlearner import TerminalDistanceRAPolicy, ReuseRAPolicyDecorator

parser = argparse.ArgumentParser()
parser.add_argument('--host', dest='host', type=str, default='localhost',
                    help='The environment host to connect to')
parser.add_argument('--port', dest='port', type=int, default=2851,
                    help='The environment port to connect to')
# Atari memory capacity was 1M
# Simulation is highly repetitive, that's why we pick a smaller number by default
parser.add_argument('--memory-capacity', dest='memory_capacity', type=int, default=50000,
                    help='The size of the memory to hold past experiences')
parser.add_argument('--image-size', dest='image_size', type=int, default=64,
                    help='The size of the (square) images to request from the environment')
parser.add_argument('--rap-initial', dest='random_action_prob_initial', type=float, default=1.0,
                    help='The initial probability of choosing a random action instead')
parser.add_argument('--rap-target', dest='random_action_prob_target', type=float, default=0.1,
                    help='The probability of choosing a random action after annealing period has passed')
# Atari annealing period was 1M
# It seems like there is not much progress being made in that manner, 10k should be enough
parser.add_argument('--rap-annealing-period', dest='random_action_prob_annealing_period',
                    type=int, default=10000,
                    help='The length of the random action annealing period in frames')
parser.add_argument('--rap-annealing', dest='use_rap_annealing', action='store_true',
                    help='Use an annealing random action policy instead of `terminal distance`')
parser.add_argument('--rap-terminal-count', dest='rap_terminal_episode_count',
                    type=int, default=20,
                    help='Use the given moving average episode count for the policy')
parser.add_argument('--rap-reuse', dest='rap_reuse_prob', type=float, default=0.8,
                    help='Enable reusing random actions with the given probability')
parser.add_argument('--batch-size', dest='batch_size',
                    type=int, default=32,
                    help='The minibatch size to use for training')
parser.add_argument('--discount', dest='discount',
                    type=float, default=0.975,
                    help='The discount to apply to future rewards (gamma)')
parser.add_argument('--episodes', dest='episodes', type=int, default=1000000,
                    help='The number of episodes to learn')
parser.add_argument('--load', dest='should_load', action='store_true',
                    help='Whether to load the model and memory')
parser.add_argument('--save', dest='should_save', action='store_true',
                    help='Whether to save the model and memory')
parser.add_argument('--all-actions', dest='action_type', action='store_const',
                    default=LeftRightAction, const=Action,
                    help='Allows the car also to decide whether to go forward or backward')
parser.add_argument('--no-training', dest='training_enabled', action='store_false',
                    help='Only drive model car, do not learn')
args = parser.parse_args()

environment = EnvironmentInterface(args.host, args.port)

if not args.use_rap_annealing:
    random_action_policy = TerminalDistanceRAPolicy(args.rap_terminal_episode_count)
else:
    random_action_policy = AnnealingRAPolicy(args.random_action_prob_initial,
                                             args.random_action_prob_target,
                                             args.random_action_prob_annealing_period)

if args.rap_reuse_prob:
    random_action_policy = ReuseRAPolicyDecorator(random_action_policy,
                                                  args.rap_reuse_prob)

learner = QLearner(environment,
                   args.memory_capacity,
                   args.image_size,
                   random_action_policy,
                   args.batch_size,
                   args.discount,
                   args.should_load,
                   args.should_load and args.training_enabled,
                   args.should_save,
                   args.action_type,
                   models.create_vgg_like_model)

if args.training_enabled:
    print("Start training")
    learner.start_training(args.episodes)
else:
    print("Start driving (without training)")
    learner.predict()
