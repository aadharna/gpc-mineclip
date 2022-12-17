from collections import namedtuple, deque
import random

Rollout = namedtuple('Rollout',
                        ('states', 'actions'))


class ReplayMemory(object):

    def __init__(self, capacity,si_counter = 1):
        self.memory = deque([],maxlen=capacity)
        self.si_counter = si_counter
        self.average_reward = 0

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Rollout(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)