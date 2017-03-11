from collections import deque


class RunningAverage:

    def __init__(self, count: int, start_value: float):
        self.values = deque((start_value for _ in range(count)), maxlen=count)
        self.running_average = start_value

    def add(self, value: float):
        running_average_count = len(self.values)
        self.running_average -= self.values.popleft() / running_average_count
        self.values.append(value)
        self.running_average += value / running_average_count

    def get(self):
        return self.running_average

