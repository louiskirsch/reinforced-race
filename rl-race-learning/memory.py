import pickle
import random
import time
from pathlib import Path
from typing import List

from environment import State, Action


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