from abc import abstractmethod

from racepi.action import Action


class Robot:
    def move(self, action: Action):
        print("Moving by {}.".format(action))

    @abstractmethod
    def get_disqualified_and_velocity(self) -> (bool, float):
        return False, 1.0
