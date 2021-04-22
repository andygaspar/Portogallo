import copy

import torch
import numpy as np


class ReplayMemory:

    def __init__(self, max_schedule_input_size, discretisation_size, size=1000):

        size = int(size)
        self.discretisationSize = discretisation_size

        # self.states = torch.zeros((size, max_schedule_input_size))
        # self.nextStates = torch.zeros((size, max_schedule_input_size))
        # self.masks = torch.zeros((size, max_schedule_input_size))
        self.states = torch.zeros((size, 50))
        self.nextStates = torch.zeros((size, 50))
        self.masks = torch.zeros((size, 20))
        self.actions = torch.zeros((size, 20))
        self.sizes = torch.zeros(size)
        self.action_size = torch.zeros(size)
        self.rewards = torch.zeros(size)
        self.losses = torch.ones(size) * 20_000
        self.done = torch.zeros(size)
        self.idx = 0
        self.size = size
        self.current_size = 0


    def set_initial_state(self, state):
        instance_size = state.shape[0]
        self.states[self.idx, :state.shape[0]] = state


    def add_record(self, next_state, action, mask, reward, final=False):
        instance_size = next_state.shape[0]
        self.sizes[self.idx] = instance_size
        self.nextStates[self.idx, :instance_size] = next_state
        self.action_size[self.idx] = action.shape[0]
        self.actions[self.idx, :action.shape[0]] = action
        self.rewards[self.idx] = reward
        self.masks[self.idx, :mask.shape[0]] = mask

        if not final:
            self.done[self.idx] = 0
            self.idx = (self.idx + 1) % self.size
            self.current_size = min(self.current_size + 1, self.size)
            self.set_initial_state(next_state)
        else:
            self.done[self.idx] = 1
            self.idx = (self.idx + 1) % self.size
            self.current_size = min(self.current_size + 1, self.size)


    def sample(self, sample_size):
        losses = self.losses.squeeze().numpy()[:self.current_size]
        p = losses / losses.sum()
        sample_idxs = np.random.choice(range(self.current_size), sample_size, p=p)
        return self.states[sample_idxs], self.nextStates[sample_idxs], self.masks[sample_idxs],\
               self.actions[sample_idxs], self.rewards[sample_idxs], self.done[sample_idxs]

    def get_episode(self):
        return

    def update_losses(self, idxs, loss):
        self.losses[idxs] = loss.item()
