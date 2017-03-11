import argparse
import random
from abc import abstractmethod
from pathlib import Path
from typing import Iterable, Any
import yaml

import numpy as np
import time
from keras.engine import Model
from keras.layers import Convolution2D, Flatten, Dense, Activation, Dropout, MaxPooling2D
from keras.models import Sequential, load_model

from environment import Action, LeftRightAction, State, EnvironmentInterface
from memory import Experience, Memory
from utils import RunningAverage


class RandomActionPolicy:

    def epoch_started(self):
        pass

    def epoch_ended(self):
        pass

    @abstractmethod
    def get_probability(self, frame: int) -> float:
        raise NotImplementedError('Subclass RandomActionPolicy')

    def sample_action(self, action_type: Any):
        return action_type.random()


class AnnealingRAPolicy(RandomActionPolicy):

    def __init__(self, initial: float, target: float, annealing_period: int):
        self.initial = initial
        self.target = target
        self.difference = initial - target
        self.annealing_period = annealing_period

    def get_probability(self, frame: int) -> float:
        if frame >= self.annealing_period:
            return self.target
        return self.initial - (frame / self.annealing_period) * self.difference


class TerminalDistanceRAPolicy(RandomActionPolicy):

    def __init__(self, running_average_count: int):
        self.running_average = RunningAverage(running_average_count, start_value=10000)
        self.epoch_started_time = None
        self.last_epoch_duration = 0.0

    def epoch_started(self):
        self.epoch_started_time = time.perf_counter()

    def epoch_ended(self):
        self.last_epoch_duration = time.perf_counter() - self.epoch_started_time
        self.running_average.add(self.last_epoch_duration)

    def get_probability(self, frame: int) -> float:
        ratio = self.last_epoch_duration / self.running_average.get()
        return min(4 ** (-ratio + 0.8), 1.0)


class ReuseRAPolicyDecorator(RandomActionPolicy):

    def __init__(self, wrapped_policy: RandomActionPolicy, reuse_prob: float):
        self.wrapped_policy = wrapped_policy
        self.reuse_prob = reuse_prob
        self.last_action = None

    def epoch_started(self):
        self.wrapped_policy.epoch_started()

    def get_probability(self, frame: int) -> float:
        return self.wrapped_policy.get_probability(frame)

    def epoch_ended(self):
        self.wrapped_policy.epoch_ended()

    def sample_action(self, action_type: Any):
        if self.last_action is not None and random.random() < self.reuse_prob:
            return self.last_action
        self.last_action = self.wrapped_policy.sample_action(action_type)
        return self.last_action


class TrainingInfo:

    INFO_FILE = 'training-info.yaml'

    def __init__(self, should_load: bool):
        file_path = Path(self.INFO_FILE)
        if should_load and file_path.is_file():
            with file_path.open() as file:
                self.data = yaml.safe_load(file)
        else:
            self.data = {
                'episode': 1,
                'frames': 0,
                'mean_training_time': 1.0
            }

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def save(self):
        with open(self.INFO_FILE, 'w') as file:
            yaml.safe_dump(self.data, file, default_flow_style=False)


class QLearner:

    MODEL_PATH = 'actionValue.model'

    def __init__(self, environment: EnvironmentInterface, memory_capacity: int, image_size: int,
                 random_action_policy: RandomActionPolicy, batch_size: int, discount: float,
                 should_load_model: bool, should_load_memory: bool, should_save: bool, action_type: Any):
        self.environment = environment
        self.random_action_policy = random_action_policy
        self.image_size = image_size
        self.batch_size = batch_size
        self.discount = discount
        self.action_type = action_type
        self.should_save = should_save
        self.training_info = TrainingInfo(should_load_model)
        self.mean_training_time = RunningAverage(1000, self.training_info['mean_training_time'])

        self.memory = Memory(memory_capacity, should_save)
        if should_load_memory:
            self.memory.load()

        if should_load_model and Path(self.MODEL_PATH).is_file():
            self.model = load_model(self.MODEL_PATH)
        else:
            self.model = self._create_model()

    def _create_model(self) -> Model:
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(self.image_size, self.image_size, 4)))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(64, 3, 3, border_mode='valid'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(self.action_type.COUNT, activation='linear'))
        model.compile(optimizer='RMSprop', loss='mse', metrics=['mean_squared_error'])

        return model

    def _predict(self, state: State) -> np.ndarray:
        # Add batch dimension
        x = np.expand_dims(state.data, axis=0)
        return self.model.predict_on_batch(x)[0]

    def _predict_multiple(self, states: Iterable[State]) -> np.ndarray:
        x = np.stack(state.data for state in states)
        return self.model.predict_on_batch(x)

    def _generate_minibatch(self) -> (np.ndarray, np.ndarray):
        batch = self.memory.random_sample(self.batch_size)

        # Estimate Q values using current model
        from_state_estimates = self._predict_multiple(experience.from_state for experience in batch)
        to_state_estimates = self._predict_multiple(experience.to_state for experience in batch)

        # Create arrays to hold input and expected output
        x = np.stack(experience.from_state.data for experience in batch)
        y = from_state_estimates

        # Reestimate y values where new reward is known
        for index, experience in enumerate(batch):
            new_y = experience.reward
            if not experience.to_state.is_terminal:
                new_y += self.discount * np.max(to_state_estimates[index])
            y[index, experience.action.get_code()] = new_y

        return x, y

    def _train_minibatch(self):
        if len(self.memory) < 1:
            return
        start = time.perf_counter()
        x, y = self._generate_minibatch()
        self.model.train_on_batch(x, y)
        end = time.perf_counter()
        self.mean_training_time.add(end - start)

    def predict(self):
        while True:
            state = self.environment.read_sensors(self.image_size, self.image_size)[0]
            while not state.is_terminal:
                action = self.action_type.from_code(np.argmax(self._predict(state)))
                self.environment.write_action(action)
                # Wait as long as we usually need to wait due to training
                time.sleep(self.training_info['mean_training_time'])
                new_state, reward = self.environment.read_sensors(self.image_size, self.image_size)
                experience = Experience(state, action, reward, new_state)
                self.memory.append_experience(experience)
                state = new_state

    def start_training(self, episodes: int):
        start_episode = self.training_info['episode']
        frames_passed = self.training_info['frames']
        for episode in range(start_episode, episodes + 1):
            self.random_action_policy.epoch_started()
            # Set initial state
            state = self.environment.read_sensors(self.image_size, self.image_size)[0]
            while not state.is_terminal:
                random_probability = self.random_action_policy.get_probability(frames_passed)
                if random.random() < random_probability:
                    action = self.random_action_policy.sample_action(self.action_type)
                else:
                    # noinspection PyTypeChecker
                    action = self.action_type.from_code(np.argmax(self._predict(state)))
                self.environment.write_action(action)
                self._train_minibatch()
                new_state, reward = self.environment.read_sensors(self.image_size, self.image_size)
                experience = Experience(state, action, reward, new_state)
                self.memory.append_experience(experience)
                state = new_state
                frames_passed += 1

                # Print status
                print('Episode {}, Total frames {}, ε={:.4f}, Action (v={:+d}, h={:+d}), Reward {}'
                      .format(episode, frames_passed, random_probability,
                              action.vertical, action.horizontal, reward), end='\r')

                # Save model after a fixed amount of frames
                if self.should_save and frames_passed % 1000 == 0:
                    self.training_info['episode'] = episode
                    self.training_info['frames'] = frames_passed
                    self.training_info['mean_training_time'] = self.mean_training_time.get()
                    self.training_info.save()
                    self.model.save(self.MODEL_PATH)

            self.random_action_policy.epoch_ended()


if __name__ == '__main__':
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
                        type=int, default=50,
                        help='Use the given moving average episode count for the policy')
    parser.add_argument('--rap-reuse', dest='rap_reuse_prob', type=float, default=None,
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

    if args.rap_reuse_prob is not None:
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
                       args.action_type)

    if args.training_enabled:
        print("Start training")
        learner.start_training(args.episodes)
    else:
        print("Start driving (without training)")
        learner.predict()
