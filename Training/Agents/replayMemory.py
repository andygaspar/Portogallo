import copy

import torch
import numpy as np
#from sklearn.preprocessing import normalize


class ReplayMemory:

    def __init__(self, max_num_flights, discretisation_size, size=1000):
        size = int(size)
        self.discretisationSize = discretisation_size
        max_size = max_num_flights * (self.discretisationSize + 7)
        self.episodeStates = None
        self.episodeMask = None
        self.episodeActions = None
        self.episodeRewards = None
        self.episodePartialRewards = None
        self.episode_idx = None
        self.episodeNumFlights = None
        self.episodeNumAirlines = None

        self.currentMask = None

        self.states = torch.zeros((size, max_size))
        self.nextStates = torch.zeros((size, max_size))
        self.masks = torch.zeros((size, max_size))
        self.actions = torch.zeros((size, 50))
        self.num_airlines = torch.zeros(size)
        self.sizes = torch.zeros(size)
        self.action_size = torch.zeros(size)
        self.rewards = torch.zeros(size)
        self.losses = torch.ones(size) * 20_000
        self.done = torch.zeros(size)
        self.idx = 0
        self.size = size
        self.current_size = 0

    def init_episode(self, act_in_episode, num_flights, num_airlines, num_trades):
        self.episodeStates = torch.zeros((act_in_episode, (self.discretisationSize + num_airlines) * num_flights +
                                          num_flights))
        self.episodeActions = torch.zeros((act_in_episode, num_flights))
        self.episodeMask = torch.zeros((act_in_episode, num_flights))
        self.episodeRewards = torch.zeros(act_in_episode)
        self.episodePartialRewards = torch.zeros(act_in_episode)
        self.episodeNumFlights = num_flights
        self.episodeNumAirlines = num_airlines
        self.episode_idx = 0

    def set_initial_state(self, state, mask, initial=True):
        instance_size = state.shape[0]
        self.episodeStates[self.episode_idx, : instance_size] = state.clone()
        self.states[self.idx, :state.shape[0]] = state
        self.current_size = min(self.current_size + 1, self.size)
        if initial:
            self.currentMask = mask.clone()

    def add_record(self, next_state, action, mask, reward, actions_in_episode=0, final=False):
        instance_size = next_state.shape[0]
        self.sizes[self.idx] = instance_size
        self.nextStates[self.idx, :instance_size] = next_state
        self.action_size[self.idx] = action.shape[0]
        self.actions[self.idx, :action.shape[0]] = action
        self.rewards[self.idx] = reward
        self.done[self.idx] = 0
        self.masks[self.idx, :mask.shape[0]] = mask

        self.episodeMask[self.episode_idx] = self.currentMask.clone()
        self.episodeActions[self.episode_idx] = action.clone()
        self.episodePartialRewards[self.episode_idx] = reward

        self.idx = (self.idx + 1) % self.size
        self.episode_idx += 1

        if not final:
            self.set_initial_state(next_state, mask, initial=False)
        else:
            self.episodePartialRewards[self.episode_idx-1] = self.episodePartialRewards[self.episode_idx-2].item()
            for i in range(1, actions_in_episode+1):
                self.rewards[self.idx-i] = reward
                self.episodeRewards[self.episode_idx - i] = reward
            self.done[self.idx] = 1

    def sample(self, sample_size):
        losses = self.losses.squeeze().numpy()[:self.current_size]
        p = losses/losses.sum()
        sample_idxs = np.random.choice(range(self.current_size), sample_size, p=p)
        return (self.states[sample_idxs], self.nextStates[sample_idxs], self.masks[sample_idxs],
                self.actions[sample_idxs], self.rewards[sample_idxs], self.done[sample_idxs]), sample_idxs

    def get_last_episode(self, num_actions):
        return self.episodeStates[:num_actions], self.episodeActions[:num_actions], \
               self.episodePartialRewards[:num_actions], self.episodeRewards[:num_actions],\
               self.episodeMask[:num_actions], self.episodeNumFlights, self.episodeNumAirlines

    def update_losses(self, idxs, loss):
        self.losses[idxs] = loss.item()

    def partial_reward(self, len_step, partial_reward):
        for i in range(1, len_step + 1):
            self.rewards[self.idx - i] = partial_reward
            self.episodePartialRewards[self.episode_idx - i] = partial_reward

    def end_short_episode(self, reward, instance_size, actions_in_episode):
        self.nextStates[self.idx, :instance_size] = -torch.ones(instance_size) * 1
        for i in range(1, actions_in_episode + 1):
            self.rewards[self.idx - i] = reward
            self.episodeRewards[self.episode_idx - i] = reward
        self.done[self.idx] = 1


class ReplayMemoryFucker:

    def __init__(self, max_schedule_input_size, discretisation_size, size=1000):

        size = int(size)
        self.discretisationSize = discretisation_size
        self.episodeStates = None
        self.episodeActions = None
        self.episodeRewards = None
        self.episode_idx = None

        self.states = torch.zeros((size, max_schedule_input_size))
        self.nextStates = torch.zeros((size, max_schedule_input_size))
        self.masks = torch.zeros((size, max_schedule_input_size))
        self.actions = torch.zeros((size, 50))
        self.num_airlines = torch.zeros(size)
        self.sizes = torch.zeros(size)
        self.action_size = torch.zeros(size)
        self.rewards = torch.zeros(size)
        self.losses = torch.ones(size) * 20_000
        self.done = torch.zeros(size)
        self.idx = 0
        self.size = size
        self.current_size = 0

        self.probs = None

    def init_episode(self, act_in_episode, num_flights, num_airlines, num_trades):
        self.episodeStates = torch.zeros(
            (act_in_episode, (self.discretisationSize + num_airlines + num_trades + 1) * num_flights))
        self.episodeActions = torch.zeros((act_in_episode, num_flights))
        self.episodeRewards = torch.zeros(act_in_episode)
        self.probs = torch.zeros(act_in_episode)

        self.episode_idx = 0

    def set_initial_state(self, state):
        instance_size = state.shape[0]
        self.episodeStates[self.episode_idx, : instance_size] = state
        self.states[self.idx, :state.shape[0]] = state
        self.current_size = min(self.current_size + 1, self.size)

    def add_record(self, next_state, action, mask, reward, prob, actions_in_episode=0, final=False):
        instance_size = next_state.shape[0]
        self.sizes[self.idx] = instance_size
        self.nextStates[self.idx, :instance_size] = next_state
        self.action_size[self.idx] = action.shape[0]
        self.actions[self.idx, :action.shape[0]] = action
        self.rewards[self.idx] = reward
        self.done[self.idx] = 0
        self.masks[self.idx, :mask.shape[0]] = mask
        self.episodeActions[self.episode_idx] = action.clone()
        self.idx = (self.idx + 1) % self.size

        self.probs[self.episode_idx] = prob

        self.episode_idx += 1

        if not final:
            self.set_initial_state(next_state)
        else:
            self.rewards[self.idx] = reward
            self.episodeRewards = torch.ones(actions_in_episode) * reward
            self.done[self.idx] = 1

    def sample(self, sample_size):
        losses = self.losses.squeeze().numpy()[:self.current_size]
        p = losses / losses.sum()
        sample_idxs = np.random.choice(range(self.current_size), sample_size, p=p)
        return (self.states[sample_idxs], self.nextStates[sample_idxs], self.masks[sample_idxs],
                self.actions[sample_idxs], self.rewards[sample_idxs], self.done[sample_idxs]), sample_idxs

    def get_last_episode(self):
        return self.episodeStates, self.episodeActions, self.episodeRewards, self.probs

    def update_losses(self, idxs, loss):
        self.losses[idxs] = loss.item()
