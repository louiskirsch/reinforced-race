#!/usr/bin/python3

import argparse

from racelearning import models
from racelearning.environment import LeftRightAction, Action, EnvironmentInterface
from racelearning.memory import Memory, EmotionalMemory
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
# Atari annealing period was 1M, initial 1.0 and target 0.1
parser.add_argument('--rap-annealing', dest='rap_annealing', metavar='INITIAL TARGET ANNEALING_PERIOD',
                    type=float, nargs=3, default=[1.0, 0.0, 1000],
                    help='Use an annealing random action policy with INITIAL probability, and TARGET probability '
                         'after annealed over ANNEALING_PERIOD frames')
parser.add_argument('--rap-terminal-count', dest='rap_terminal_episode_count', type=int,
                    help='Use the given moving average episode count for the random policy instead. '
                         '20 is a good default.')
parser.add_argument('--rap-reuse', dest='rap_reuse_prob', type=float, default=0.8,
                    help='Enable reusing random actions with the given probability')
parser.add_argument('--emotional-memory', dest='emotional_memory_length', type=int, default=20,
                    help='Emotional memory that highlights previous failures '
                         'with the given length in frames per event. Set to 0 to disable.')
parser.add_argument('--batch-size', dest='batch_size',
                    type=int, default=32,
                    help='The minibatch size to use for training')
parser.add_argument('--batches-per-frame', dest='batches_per_frame',
                    type=int, default=None,
                    help='The number of minibatch training iterations per frame')
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
parser.add_argument('--atari', dest='model_creator', action='store_const',
                    const=models.create_atari_model,
                    help='Use the original atari model for predicting returns, this is the default')
parser.add_argument('--vgg', dest='model_creator', action='store_const',
                    const=models.create_vgg_like_model,
                    help='Use a VGG like model for predicting returns')
parser.set_defaults(model_creator=models.create_atari_model)
args = parser.parse_args()

environment = EnvironmentInterface(args.host, args.port)

if args.rap_terminal_episode_count:
    random_action_policy = TerminalDistanceRAPolicy(args.rap_terminal_episode_count)
else:
    initial, target, period = args.rap_annealing
    random_action_policy = AnnealingRAPolicy(initial, target, period)

if args.rap_reuse_prob:
    random_action_policy = ReuseRAPolicyDecorator(random_action_policy,
                                                  args.rap_reuse_prob)


if args.emotional_memory_length:
    memory = EmotionalMemory(args.memory_capacity, args.should_save, args.emotional_memory_length)
else:
    memory = Memory(args.memory_capacity, args.should_save)

if args.should_load and args.training_enabled:
    memory.load()

learner = QLearner(environment,
                   memory,
                   args.image_size,
                   random_action_policy,
                   args.batch_size,
                   args.discount,
                   args.should_load,
                   args.should_save,
                   args.action_type,
                   args.model_creator,
                   args.batches_per_frame)

if args.training_enabled:
    print("Start training")
    learner.start_training(args.episodes)
else:
    print("Start driving (without training)")
    learner.predict()
