import socket
from struct import pack, unpack, calcsize

import numpy as np
from typing import Any, Tuple

from racepi.action import Action
from racepi.camera import Camera
from racepi.constants import REQUEST_READ_SENSORS, REQUEST_WRITE_ACTION, DEFAULT_PORT
from racepi.robot import Robot


class Server:
    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = DEFAULT_PORT,
                 robot: Robot = Robot(),
                 camera: Camera = Camera()):
        self.camera = camera
        self.robot = robot
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((host, port))
        self.socket.listen(1)
        print("Server listening...")
        self.connection, address = self.socket.accept()
        print("Server accepted connection from {}.".format(address))

    def keep_handling_commands(self):
        while True:
            self.handle_command()

    def handle_command(self):
        command_type, = self.receive('b')
        if command_type == REQUEST_READ_SENSORS:
            print("Requesting sensors.")
            self.robot.get_disqualified_and_velocity()
            width, height = self.receive("ii")
            picture = self.camera.get_picture(width=width, height=height)

            disqualified, velocity = self.robot.get_disqualified_and_velocity()
            finished = False

            # Velocity is encoded as x * 2^16
            self.send("??i", disqualified, finished, int(velocity * 0xffff))

            for byte in list(np.reshape(picture, (width * height,))):
                self.send("B", byte)

        elif command_type == REQUEST_WRITE_ACTION:
            print("Requesting action.")
            vertical, horizontal = self.receive("ii")
            self.robot.move(Action(vertical=vertical, horizontal=horizontal))

    def receive(self, type_sequence: str) -> Tuple:
        size = calcsize(type_sequence)
        print("Waiting for structs of type {} with length {}".format(type_sequence, size))
        received_bytes = self.connection.recv(size)
        print("Received bytes {}.".format(received_bytes))
        unpacked = unpack("!" + type_sequence, received_bytes)
        print("Received {}.".format(unpacked))

        return unpacked

    def send(self, type_sequence: str, *args: Any):
        self.connection.sendall(pack("!" + type_sequence, *args))
