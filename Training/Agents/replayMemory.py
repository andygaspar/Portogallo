import copy

import torch
import numpy as np
#from sklearn.preprocessing import normalize


class ReplayMemory:

    def __init__(self, max_num_flights, size=1000):
        size = int(size)
        self.states = torch.zeros((size, max_num_flights))
        self.nextStates = torch.zeros((size, max_num_flights))
        self.masks = torch.zeros((size, max_num_flights))
        self.actions = torch.zeros((size, max_num_flights))
        self.num_airlines = torch.zeros((size, 1))
        self.sizes = torch.zeros((size, 1))
        self.rewards = torch.zeros((size, 1))
        self.losses = torch.ones((size, 1)) * 20_000
        self.done = torch.zeros((size, 1))
        self.idx = 0
        self.size = size
        self.current_size = 0

    def set_initial_state(self, state, instance_size):
        self.states[self.idx, :instance_size] = state
        self.current_size = min(self.current_size + 1, self.size)

    def add_record(self, next_state, action, mask, reward, final=False):
        instance_size = next_state.shape[0]
        self.sizes[self.idx] = instance_size
        self.nextStates[self.idx, :instance_size] = next_state
        self.actions[self.idx, :instance_size] = action
        self.rewards[self.idx] = reward
        self.done[self.idx] = 0
        self.masks[self.idx] = mask

        self.idx = (self.idx + 1) % self.size

        if not final:
            self.set_initial_state(next_state, instance_size)
        else:
            self.done[self.idx] = 1

    def sample(self, sample_size):
        losses = self.losses.squeeze().numpy()[:self.current_size]
        p = losses/losses.sum()
        sample_idxs = np.random.choice(range(self.current_size), sample_size, p=p)
        return (self.states[sample_idxs], self.nextStates[sample_idxs], self.masks[sample_idxs],
                self.actions[sample_idxs], self.rewards[sample_idxs], self.done[sample_idxs]), sample_idxs

    def update_losses(self, idxs, loss):
        self.losses[idxs] = loss.item()
