from collections import deque
import random


class ReplayMemory:
    #creating replay memory -  experience buffer
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)

    #appending experience in memory
    def append(self, new_experience):
        self.memory.append(new_experience)

    # returning random  samples
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    # calcualting length
    def __len__(self):
     return len(self.memory)
