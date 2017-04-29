from racepi.action import Action
from racepi.enviroment_mock import EnvironmentInterfaceMock

if __name__ == '__main__':
    mock = EnvironmentInterfaceMock(host="pi")
    result = mock.read_sensors(width=30, height=30)
    print(result)
    mock.write_action(Action(-1, 1))