import copy

import torch
import numpy as np


class ReplayMemory:

    def __init__(self, action_size, state_size, size=1000):
        size = int(size)
        self.states = torch.zeros((size, state_size))
        self.nextStates = torch.zeros((size, state_size))
        self.masks = torch.zeros((size, action_size))
        self.actions = torch.zeros((size, action_size))
        self.rewards = torch.zeros((size, 1))
        self.done = torch.zeros((size, 1))
        self.idx = 0
        self.size = size
        self.current_size = 0

    def set_initial_state(self, state):
        self.states[self.idx] = state
        self.current_size = min(self.current_size + 1, self.size)

    def add_record(self, next_state, action, reward, mask, done=0, initial=False):
        self.nextStates[self.idx] = next_state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.done[self.idx] = done
        self.masks[self.idx] = torch.tensor([-1 if mask[i] == 0 else 0 for i in range(len(mask))])

        self.idx = (self.idx + 1) % self.size

        if initial:
            self.set_initial_state(next_state)

    def sample(self, sample_size):
        sample_idxs = np.random.choice(self.current_size, sample_size)
        return self.states[sample_idxs], self.nextStates[sample_idxs], self.masks[sample_idxs], self.actions[sample_idxs], \
               self.rewards[sample_idxs], self.done[sample_idxs]
