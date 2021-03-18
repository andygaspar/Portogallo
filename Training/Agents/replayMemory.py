import copy

import torch
import numpy as np
from sklearn.preprocessing import normalize


class ReplayMemory:

    def __init__(self, action_size, state_size, size=1000):
        size = int(size)
        state_size = (15 + 2 + 4)*16 + 24*6 + 24
        self.states = torch.zeros((size, state_size))
        self.nextStates = torch.zeros((size, state_size))
        self.masks = torch.zeros((size, action_size))
        self.actions = torch.zeros((size, action_size))
        self.rewards = torch.zeros((size, 1))
        self.losses = torch.ones((size, 1)) * 20_000
        self.done = torch.zeros((size, 1))
        self.idx = 0
        self.size = size
        self.current_size = 0

    def set_initial_state(self, state):
        self.states[self.idx] = self.redef_state(state)
        self.current_size = min(self.current_size + 1, self.size)

    def add_record(self, next_state, action, reward, mask, done=0, initial=False):
        self.nextStates[self.idx] = self.redef_state(next_state)
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.done[self.idx] = done
        self.masks[self.idx] = mask

        self.idx = (self.idx + 1) % self.size

        if initial:
            self.set_initial_state(next_state)

    def sample(self, sample_size):
        losses = self.losses.squeeze().numpy()[:self.current_size]
        p = losses/losses.sum()
        sample_idxs = np.random.choice(range(self.current_size), sample_size, p=p)
        sample_idxs = np.random.choice(sample_idxs, 2*self.current_size//3)
        non_zero_rewards_idx = torch.nonzero(self.rewards)[:, 0]
        non_zero_rewards = self.rewards[non_zero_rewards_idx]
        sample_idxs = np.append(sample_idxs, non_zero_rewards_idx[
            torch.argsort(non_zero_rewards, dim=0, descending=True)][:(self.current_size-2*self.current_size//3)])
        return (self.states[sample_idxs], self.nextStates[sample_idxs], self.masks[sample_idxs],
                self.actions[sample_idxs], self.rewards[sample_idxs], self.done[sample_idxs]), sample_idxs

    def update_losses(self, idxs, loss):
        self.losses[idxs] = loss.item()

    def redef_state(self, state):

        flights, trades, current_trade = torch.split(state, [(4+15+2)*16,
                                                             20 * 6,
                                                             20], dim=-1)

        trades = torch.split(trades, [20 for _ in range(6)], dim=-1)

        new_trades = []
        for trade in trades:
            new_trades.append(torch.zeros(24))

            if len(torch.nonzero(trade[:4])) > 0:

                first_idx = torch.nonzero(trade[:4])[0].item()
                first_idx = first_idx * 6
                first_idx = first_idx + torch.nonzero(trade[4:10])[0].item()

                sec_idx = torch.nonzero(trade[10:14])[0].item()
                sec_idx = sec_idx * 6
                sec_idx = sec_idx + torch.nonzero(trade[14:20])[0].item()

                new_trades[-1][first_idx] = 1
                new_trades[-1][sec_idx] = 1

        trades = torch.cat(new_trades, dim=-1)

        curr = torch.zeros(24)
        if len(torch.nonzero(current_trade[:4])) > 0 and len(torch.nonzero(current_trade[4:10])) > 0:
            first_idx = torch.nonzero(current_trade[:4])[0].item()
            first_idx = first_idx * 6
            first_idx = first_idx + torch.nonzero(current_trade[4:10])[0].item()
            curr[first_idx] = 1

            if len(torch.nonzero(current_trade[10:14])) > 0 and len(torch.nonzero(current_trade[14:20])) > 0:
                sec_idx = torch.nonzero(current_trade[10:14])[0].item()
                sec_idx = sec_idx * 6
                sec_idx = sec_idx + torch.nonzero(current_trade[14:20])[0].item()
                curr[sec_idx] = 1

        final = torch.cat([flights, trades, curr])

        return final


