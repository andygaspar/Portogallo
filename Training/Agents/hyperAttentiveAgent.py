import copy
from typing import List

import numpy as np
import torch
from torch import optim
from Training.Agents.replayMemory import ReplayMemory
from Training.Agents import flAgent, airAgent, attentionAgent
from Training.masker import Masker


class AttentiveHyperAgent:

    def __init__(self, num_flight_types, num_airlines, num_flights, num_trades, num_combs, weight_decay,
                 trainings_per_step=10, batch_size=200, memory_size=10000, train_mode=False):

        ETA_info_size = 1
        time_info_size = 1
        self.singleTradeSize = (num_airlines + num_combs) * 2  # 2 as we are dealing with couples
        self.currentTradeSize = self.singleTradeSize
        self.numAirlines = num_airlines
        schedule_entry_size = num_flight_types + ETA_info_size + time_info_size + num_airlines
        input_size = schedule_entry_size * num_flights + num_trades * self.singleTradeSize + self.currentTradeSize

        self.numFlights = 16  # da sistemare

        self.weightDecay = weight_decay

        l_rate = 1e-3
        hidden_dim = 64

        self.network = attentionAgent.attentionNet(num_airlines, hidden_dim, schedule_entry_size, self.singleTradeSize,
                                                   num_flights, num_trades, l_rate, weight_decay=1e-4)

        self.trainMode = train_mode
        self.trainingsPerStep = trainings_per_step
        self.batchSize = batch_size

        self.replayMemory = ReplayMemory(self.numAirlines, input_size, size=memory_size)

    def pick_flight(self, state, eps, masker: Masker):
        actions = torch.zeros_like(masker.flMask)
        if self.trainMode and np.random.rand() < eps:
            action = np.random.choice([i for i in range(len(masker.flMask)) if round(masker.flMask[i].item()) == 1])
            masker.fl_action(action)
            actions[action] = 1
            return actions
        scores = self.network.pick_action(state)
        scores[masker.flMask == 0] = -float('inf')
        action = torch.argmax(scores)
        actions[action] = 1
        masker.fl_action(action.item())
        return actions

    def step(self, schedule: torch.tensor, trade_list: torch.tensor, eps, len_step=4,
             masker=None, last_step=True, train=True):

        current_trade = torch.zeros(self.numFlights)
        state = torch.cat([schedule, trade_list, current_trade], dim=-1)
        self.replayMemory.set_initial_state(state)

        for _ in range(len_step - 1):
            action = self.pick_flight(state, eps, masker)
            current_trade += action
            state[-self.currentTradeSize:] = current_trade
            self.replayMemory.add_record(next_state=state, action=action, mask=masker.airMask, reward=0)

        action = self.pick_flight(state, eps, masker)
        current_trade += action
        state[-self.numFlights:] = current_trade

        if not last_step:
            state[-self.numFlights:] = current_trade
            self.replayMemory.add_record(next_state=state, action=action, mask=masker.airMask, reward=0)
            return current_trade
        else:
            last_state = torch.ones_like(state) * -1
            return current_trade, last_state

    def assign_end_episode_reward(self, last_state, action, mask, shared_reward):
        self.replayMemory.add_record(next_state=last_state, action=action, mask=mask,
                                     reward=shared_reward, final=True)

    def train(self):
        for i in range(self.trainingsPerStep):
            batch, idxs = self.replayMemory.sample(self.batchSize)
            loss = self.network.update_weights(batch)
            self.replayMemory.update_losses(idxs, loss)
