import argparse
import random
from pathlib import Path
from typing import Iterable, Any

import numpy as np
from keras.engine import Model
from keras.layers import Convolution2D, Flatten, Dense, Activation, Dropout, MaxPooling2D
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop, SGD

from environment import Action, LeftRightAction, State, EnvironmentInterface
from memory import Experience, Memory


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
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='mse', metrics=['mean_squared_error'])

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
    # Atari memory capacity was 1M
    # Simulation is highly repetitive, that's why we pick a smaller number by default
    parser.add_argument('--memory-capacity', dest='memory_capacity', type=int, default=50000,
                        help='The size of the memory to hold past experiences')
    parser.add_argument('--image-size', dest='image_size', type=int, default=64,
                        help='The size of the (square) images to request from the environment')
    parser.add_argument('--rap-initial', dest='random_action_prob_initial', type=float, default=1.0,
                        help='The initial probability of choosing a random action instead')
    # Atari minimum random action probability was 10%
    # That leads to early fails within the simulation stopping us to get to later stages, that's why we pick 1%
    parser.add_argument('--rap-target', dest='random_action_prob_target', type=float, default=0.01,
                        help='The probability of choosing a random action after annealing period has passed')
    # Atari annealing period was 1M
    # It seems like there is not much progress being made in that manner, 10k should be enough
    parser.add_argument('--rap-annealing-period', dest='random_action_prob_annealing_period',
                        type=int, default=10000,
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
