import random
import signal
import time
from abc import abstractmethod
from pathlib import Path
from typing import Iterable, Any, Callable

import numpy as np
import sys
import yaml
from keras.engine import Model
from keras.layers import Convolution2D, Flatten, Dense, Activation, Dropout, MaxPooling2D
from keras.models import Sequential, load_model
from racelearning.memory import Experience, Memory
from racelearning.utils import RunningAverage

from racelearning.environment import State, EnvironmentInterface, StateAssembler


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
                'mean_training_time': 1.0,
                'batches_per_frame': 1
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

    def __init__(self, environment: EnvironmentInterface, memory: Memory, image_size: int,
                 random_action_policy: RandomActionPolicy, batch_size: int, discount: float,
                 should_load_model: bool, should_save: bool, action_type: Any,
                 create_model: Callable[[Any, int], Model], batches_per_frame: int):
        self.environment = environment
        self.random_action_policy = random_action_policy
        self.memory = memory
        self.image_size = image_size
        self.batch_size = batch_size
        self.discount = discount
        self.action_type = action_type
        self.should_save = should_save
        self.should_exit = False
        self.default_sigint_handler = signal.getsignal(signal.SIGINT)
        self.training_info = TrainingInfo(should_load_model)
        self.mean_training_time = RunningAverage(1000, self.training_info['mean_training_time'])
        if batches_per_frame:
            self.training_info['batches_per_frame'] = batches_per_frame

        if should_load_model and Path(self.MODEL_PATH).is_file():
            self.model = load_model(self.MODEL_PATH)
        else:
            self.model = create_model((self.image_size, self.image_size, StateAssembler.FRAME_COUNT),
                                      action_type.COUNT)

    def stop(self, sig, frame):
        print('Exiting...')
        self.should_exit = True

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
        signal.signal(signal.SIGINT, self.stop)
        while True:
            state = self.environment.read_sensors(self.image_size, self.image_size)[0]
            while not state.is_terminal:
                action = self.action_type.from_code(np.argmax(self._predict(state)))
                self.environment.write_action(action)
                # Wait as long as we usually need to wait due to training
                time.sleep(self.training_info['batches_per_frame'] *
                           self.training_info['mean_training_time'])
                new_state, reward = self.environment.read_sensors(self.image_size, self.image_size)
                experience = Experience(state, action, reward, new_state)
                self.memory.append_experience(experience)
                state = new_state

                if self.should_exit:
                    sys.exit(0)

    def start_training(self, episodes: int):
        signal.signal(signal.SIGINT, self.stop)
        start_episode = self.training_info['episode']
        frames_passed = self.training_info['frames']
        for episode in range(start_episode, episodes + 1):
            self.random_action_policy.epoch_started()
            # Set initial state
            state = self.environment.read_sensors(self.image_size, self.image_size)[0]
            episode_start_time = time.time()
            while not state.is_terminal:
                random_probability = self.random_action_policy.get_probability(frames_passed)
                if random.random() < random_probability:
                    action = self.random_action_policy.sample_action(self.action_type)
                else:
                    # noinspection PyTypeChecker
                    action = self.action_type.from_code(np.argmax(self._predict(state)))
                self.environment.write_action(action)
                for _ in range(self.training_info['batches_per_frame']):
                    self._train_minibatch()
                new_state, reward = self.environment.read_sensors(self.image_size, self.image_size)
                experience = Experience(state, action, reward, new_state)
                self.memory.append_experience(experience)

                if new_state.is_terminal:
                    self.memory.report_failure()

                state = new_state
                frames_passed += 1

                # Print status
                time_since_failure = time.time() - episode_start_time
                print('Episode {}, Total frames {}, ε={:.4f}, Action (v={:+d}, h={:+d}), Reward {:.4f}, '
                      '{:.0f}s since failure'
                      .format(episode, frames_passed, random_probability,
                              action.vertical, action.horizontal, reward,
                              time_since_failure), end='\r')

                # Save model after a fixed amount of frames
                if self.should_save and frames_passed % 1000 == 0:
                    self.training_info['episode'] = episode
                    self.training_info['frames'] = frames_passed
                    self.training_info['mean_training_time'] = self.mean_training_time.get()
                    self.training_info.save()
                    self.model.save(self.MODEL_PATH)

                if self.should_exit:
                    sys.exit(0)

            self.random_action_policy.epoch_ended()
        signal.signal(signal.SIGINT, self.default_sigint_handler)

