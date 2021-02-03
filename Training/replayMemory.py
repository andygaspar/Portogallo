import copy

import torch


class ReplayMemory:

    def __init__(self, action_size= 0):
        self.states = []
        self.nextStates = []
        self.actions = []
        self.rewards = []
        self.initialState = None

    def set_initial_state(self, state):
        self.initialState = copy.deepcopy(state)

    def add_record(self, next_state, action, reward, initial=False):
        if initial:
            self.states.append(self.initialState)
        else:
            self.states.append(copy.deepcopy(self.nextStates[-1]))
        self.nextStates.append(copy.deepcopy(next_state))
        self.actions.append(copy.deepcopy(action))
        self.rewards.append(torch.tensor(reward))