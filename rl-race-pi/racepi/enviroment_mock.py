import socket
from struct import pack, unpack

import numpy as np
from numpy import ndarray

from racepi.action import Action
from racepi.constants import REQUEST_READ_SENSORS, REQUEST_WRITE_ACTION, DEFAULT_PORT


class EnvironmentInterfaceMock:
    def __init__(self, host: str, port: int = DEFAULT_PORT):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))

    def read_sensors(self, width: int, height: int) -> (bool, bool, float, ndarray):
        request = pack('!bii', REQUEST_READ_SENSORS, width, height)
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

        return disqualified, finished, velocity, camera_image

    def write_action(self, action: Action):
        request = pack('!bii', REQUEST_WRITE_ACTION, action.vertical, action.horizontal)
        self.socket.sendall(request)
