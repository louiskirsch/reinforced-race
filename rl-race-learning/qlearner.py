import argparse
from pathlib import Path
from struct import pack, unpack
from typing import List, Iterable, Any

import random
import socket
import math
import pickle
import numpy as np

import time
from keras.engine import Model
from keras.layers import Convolution2D, Flatten, Dense
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop


class Action:

    COUNT = 9

    def __init__(self, vertical: int, horizontal: int):
        if vertical > 1 or vertical < -1:
            raise ValueError('Vertical must be -1, 0 or 1')
        if horizontal > 1 or horizontal < -1:
            raise ValueError('Horizontal must be -1, 0 or 1')
        self.vertical = vertical
        self.horizontal = horizontal

    @classmethod
    def from_code(cls, code: int):
        return cls(code // 3 - 1, code % 3 - 1)

    @classmethod
    def random(cls):
        return cls(random.randrange(3) - 1, random.randrange(3) - 1)

    def get_code(self) -> int:
        return (self.vertical + 1) * 3 + self.horizontal + 1


class LeftRightAction(Action):

    COUNT = 3

    def __init__(self, horizontal: int):
        super().__init__(1, horizontal)

    @classmethod
    def from_code(cls, code: int):
        return cls(code - 1)

    @classmethod
    def random(cls):
        return cls(random.randrange(3) - 1)

    def get_code(self) -> int:
        return self.horizontal + 1


class State:

    def __init__(self, data: np.ndarray, is_terminal: bool):
        # TODO normalize data?
        # Convert to 0 - 1 ranges
        data = data.astype(np.float32) / 255
        # Add channel dimension
        self.data = np.expand_dims(data, axis=2)
        self.is_terminal = is_terminal


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class EnvironmentInterface:

    REQUEST_READ_SENSORS = 1
    REQUEST_WRITE_ACTION = 2

    def __init__(self, host: str, port: int):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))

    @staticmethod
    def _calc_reward(disqualified: bool, finished: bool, velocity: float) -> float:
        if disqualified:
            return -1e+4
        if finished:
            return 0
        # If velocity = 0 then reward = -1 otherwise get closer to 0
        return sigmoid(velocity / 2) * 2 - 2

    def read_sensors(self, width: int, height: int) -> (State, int):
        request = pack('!bii', self.REQUEST_READ_SENSORS, width, height)
        self.socket.sendall(request)

        # Response size: disqualified, finished, velocity, camera_image
        response_size = width * height + 2 + 4
        response_buffer = bytes()
        while len(response_buffer) < response_size:
            response_buffer += self.socket.recv(response_size - len(response_buffer))

        disqualified, finished, velocity = unpack('!??i', response_buffer[:6])
        # Velocity is encoded as x * 2^16
        velocity /= 0xffff
        camera_image = np.frombuffer(response_buffer[6:], dtype=np.uint8)
        camera_image = np.reshape(camera_image, (height, width), order='C')

        reward = self._calc_reward(disqualified, finished, velocity)

        return State(camera_image, disqualified or finished), reward

    def write_action(self, action: Action):
        request = pack('!bii', self.REQUEST_WRITE_ACTION, action.vertical, action.horizontal)
        self.socket.sendall(request)


class Experience:

    OUTPUT_DIR = 'experiences'
    FILE_PREFIX = 'experience-'
    FILE_SUFFIX = '.pickle'

    def __init__(self, from_state: State, action: Action, reward: float, to_state: State):
        self.from_state = from_state
        self.action = action
        self.reward = reward
        self.to_state = to_state
        self.store_path = None

    @classmethod
    def load_many(cls, count: int):
        experiences = []
        directory = Path(cls.OUTPUT_DIR)

        if not directory.is_dir():
            return experiences

        def file_date(fp: Path):
            return int(fp.stem[len(cls.FILE_PREFIX):])

        file_paths = sorted(directory.glob('{}*{}'.format(cls.FILE_PREFIX, cls.FILE_SUFFIX)),
                            key=file_date)

        for file_path in file_paths[-count:]:
            with file_path.open('rb') as file:
                experiences.append(pickle.load(file))

        return experiences

    def delete(self):
        if self.store_path is not None and self.store_path.is_file():
            self.store_path.unlink()

    def save(self):
        Path(self.OUTPUT_DIR).mkdir(exist_ok=True)
        path = Path('{}/{}{:d}{}'.format(self.OUTPUT_DIR, self.FILE_PREFIX, int(time.time() * 1000), self.FILE_SUFFIX))
        with path.open('wb') as file:
            pickle.dump(self, file)
        self.store_path = path


class Memory:

    def __init__(self, capacity: int, preserve_to_disk: bool):
        self.experiences = []
        self.capacity = capacity
        self.write_position = 0
        self.preserve_to_disk = preserve_to_disk

    def load(self):
        self.experiences = Experience.load_many(self.capacity)
        self.write_position = len(self.experiences)

    def append_experience(self, experience: Experience):
        if self.write_position >= self.capacity:
            self.write_position = 0

        if self.write_position >= len(self.experiences):
            self.experiences.append(experience)
        else:
            old_experience = self.experiences[self.write_position]
            if self.preserve_to_disk:
                old_experience.delete()
            self.experiences[self.write_position] = experience

        if self.preserve_to_disk:
            experience.save()
        self.write_position += 1

    def __len__(self):
        return len(self.experiences)

    def random_sample(self, batch_size: int) -> List[Experience]:
        if batch_size < len(self.experiences):
            return random.sample(self.experiences, batch_size)
        return self.experiences


class RandomActionPolicy:

    def __init__(self, initial: float, target: float, annealing_period: int):
        self.initial = initial
        self.target = target
        self.difference = initial - target
        self.annealing_period = annealing_period

    def get_probability(self, frame: int) -> float:
        if frame >= self.annealing_period:
            return self.target
        return self.initial - (frame / self.annealing_period) * self.difference


class QLearner:

    MODEL_PATH = 'actionValue.model'

    def __init__(self, environment: EnvironmentInterface, memory_capacity: int, image_size: int,
                 random_action_policy: RandomActionPolicy, batch_size: int, discount: float,
                 load_memory: bool, save_memory: bool, action_type: Any, min_memory_training: int):
        self.environment = environment
        self.random_action_policy = random_action_policy
        self.image_size = image_size
        self.batch_size = batch_size
        self.discount = discount
        self.action_type = action_type
        self.min_memory_training = min_memory_training

        self.memory = Memory(memory_capacity, save_memory)
        if load_memory:
            self.memory.load()

        if Path(self.MODEL_PATH).is_file():
            self.model = load_model(self.MODEL_PATH)
        else:
            self.model = self._create_model()

    def _create_model(self) -> Model:
        model = Sequential()
        model.add(Convolution2D(16, 8, 8, activation='relu', border_mode='same',
                                input_shape=(self.image_size, self.image_size, 1), subsample=(4, 4)))
        model.add(Convolution2D(32, 4, 4, activation='relu', border_mode='same', subsample=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_type.COUNT, activation='linear'))
        model.compile(optimizer=RMSprop(), loss='mse', metrics=['mean_squared_error'])
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
        if len(self.memory) < self.min_memory_training:
            return
        x, y = self._generate_minibatch()
        self.model.train_on_batch(x, y)

    def predict(self):
        while True:
            state = self.environment.read_sensors(self.image_size, self.image_size)[0]
            while not state.is_terminal:
                action = self.action_type.from_code(np.argmax(self._predict(state)))
                self.environment.write_action(action)
                new_state, reward = self.environment.read_sensors(self.image_size, self.image_size)
                experience = Experience(state, action, reward, new_state)
                self.memory.append_experience(experience)
                state = new_state

    def start_training(self, episodes: int):
        frames_passed = 0
        for episode in range(1, episodes + 1):
            print('Running episode {}'.format(episode))
            # Set initial state
            state = self.environment.read_sensors(self.image_size, self.image_size)[0]
            while not state.is_terminal:
                if random.random() < self.random_action_policy.get_probability(frames_passed):
                    action = self.action_type.random()
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
                if frames_passed % 1000 == 0:
                    print('Completed {} frames'.format(frames_passed))
                    self.model.save(self.MODEL_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', dest='host', type=str, default='localhost',
                        help='The environment host to connect to')
    parser.add_argument('--port', dest='port', type=int, default=2851,
                        help='The environment port to connect to')
    parser.add_argument('--memory-capacity', dest='memory_capacity', type=int, default=1000000,
                        help='The size of the memory to hold past experiences')
    parser.add_argument('--image-size', dest='image_size', type=int, default=64,
                        help='The size of the (square) images to request from the environment')
    parser.add_argument('--rap-initial', dest='random_action_prob_initial', type=float, default=1.0,
                        help='The initial probability of choosing a random action instead')
    parser.add_argument('--rap-target', dest='random_action_prob_target', type=float, default=0.1,
                        help='The probability of choosing a random action after annealing period has passed')
    parser.add_argument('--rap-annealing-period', dest='random_action_prob_annealing_period',
                        type=int, default=1000000,
                        help='The length of the random action annealing period in frames')
    parser.add_argument('--batch-size', dest='batch_size',
                        type=int, default=32,
                        help='The minibatch size to use for training')
    parser.add_argument('--discount', dest='discount',
                        type=float, default=0.975,
                        help='The discount to apply to future rewards (gamma)')
    parser.add_argument('--episodes', dest='episodes', type=int, default=1000000,
                        help='The number of episodes to learn')
    parser.add_argument('--load-memory', dest='load_memory', action='store_true',
                        help='Whether to load stored memory')
    parser.add_argument('--save-memory', dest='save_memory', action='store_true',
                        help='Whether to save memory')
    parser.add_argument('--min-memory', dest='min_memory_training', type=int, default=1,
                        help='The minimum memory size before training starts')
    parser.add_argument('--always-forward', dest='action_type', action='store_const',
                        default=Action, const=LeftRightAction,
                        help='Whether the car should always move forward')
    parser.add_argument('--no-training', dest='training_enabled', action='store_false',
                        help='Only drive model car, do not learn')
    args = parser.parse_args()

    environment = EnvironmentInterface(args.host, args.port)
    random_action_policy = RandomActionPolicy(args.random_action_prob_initial,
                                              args.random_action_prob_target,
                                              args.random_action_prob_annealing_period)
    learner = QLearner(environment,
                       args.memory_capacity,
                       args.image_size,
                       random_action_policy,
                       args.batch_size,
                       args.discount,
                       args.load_memory,
                       args.save_memory,
                       args.action_type,
                       args.min_memory_training)

    if args.training_enabled:
        print("Start training")
        learner.start_training(args.episodes)
    else:
        print("Start driving (without training)")
        learner.predict()
