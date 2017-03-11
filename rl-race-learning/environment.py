import math
import random
import socket
from collections import deque
from struct import pack, unpack

import numpy as np


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
        # Convert to 0 - 1 ranges
        self.data = data.astype(np.float32) / 255
        # Z-normalize
        self.data = (self.data - np.mean(self.data)) / np.std(self.data, ddof=1)
        # Add channel dimension
        if len(self.data.shape) < 3:
            self.data = np.expand_dims(self.data, axis=2)
        self.is_terminal = is_terminal


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class StateAssembler:

    FRAME_COUNT = 4

    def __init__(self):
        self.cache = deque(maxlen=self.FRAME_COUNT)

    def assemble_next(self, camera_image: np.ndarray, is_terminal: bool) -> State:
        self.cache.append(camera_image)
        # If cache is still empty, put this image in there multiple times
        while len(self.cache) < self.FRAME_COUNT:
            self.cache.append(camera_image)
        images = np.stack(self.cache, axis=2)
        return State(images, is_terminal)


class EnvironmentInterface:

    REQUEST_READ_SENSORS = 1
    REQUEST_WRITE_ACTION = 2

    def __init__(self, host: str, port: int):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
        self.assembler = StateAssembler()

    @staticmethod
    def _calc_reward(disqualified: bool, finished: bool, velocity: float) -> float:
        if disqualified:
            return 0
        if finished:
            return 0
        return 1

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
        state = self.assembler.assemble_next(camera_image, disqualified or finished)

        return state, reward

    def write_action(self, action: Action):
        request = pack('!bii', self.REQUEST_WRITE_ACTION, action.vertical, action.horizontal)
        self.socket.sendall(request)
