import copy
from typing import List

import numpy as np
import torch
from torch import optim
from Training.replayMemory import ReplayMemory
import flAgent
import airAgent


class HyperAgent:

    def __init__(self, num_flight_types, num_airlines, num_flights, num_trades, num_combs, weight_decay, batch_size=200,
                 memory_size=10000, train_mode=False):

        ETA_info_size = 1
        time_info_size = 1
        self.singleTradeSize = (num_airlines + num_combs) * 2   # 2 as we are dealing with couples
        self.currentTradeSize = self.singleTradeSize
        self.numCombs = num_combs
        self.numAirlines = num_airlines

        input_size = (num_flight_types + ETA_info_size + time_info_size + num_airlines) * num_flights + \
                     num_trades * self.singleTradeSize + self.currentTradeSize

        self.weightDecay = weight_decay

        self.AirAgent = airAgent.AirNet(input_size, self.weightDecay, num_flight_types, num_airlines, num_flights, num_trades, num_combs)
        self.FlAgent = flAgent.FlNet(input_size, self.weightDecay, num_flight_types, num_airlines, num_flights, num_trades, num_combs)

        self.trainMode = train_mode
        self.batchSize = batch_size

        self.AirReplayMemory = ReplayMemory(self.numAirlines, input_size, size=memory_size)
        self.FlReplayMemory = ReplayMemory(self.numCombs, input_size, size=memory_size)

    def pick_air_action(self, state, eps):
        if self.trainMode and np.random.rand() < eps:
            return np.random.choice(self.numAirlines)
        scores = self.AirAgent.pick_action(state)
        action = torch.zeros_like(scores)
        action[torch.argmax(scores)] = 1
        return action

    def pick_fl_action(self, state, eps):
        if self.trainMode and np.random.rand() < eps:
            return np.random.choice(self.numCombs)
        scores = self.FlAgent.pick_action(state)
        action = torch.zeros_like(scores)
        action[torch.argmax(scores)] = 1
        return action

    def step(self, state_list: List, eps, last_step=False):
        current_trade = torch.zeros(self.singleTradeSize)
        state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)
        self.AirReplayMemory.set_initial_state(state)

        start = 0
        end = start + self.numAirlines

        air_action = self.pick_air_action(state, eps)
        current_trade[start:end] = air_action
        state[-self.currentTradeSize:] = current_trade

        start = end
        end = start + self.numCombs
        self.FlReplayMemory.set_initial_state(state)
        fl_action = self.pick_fl_action(state, eps)
        current_trade[start:end] = fl_action

        start = end
        end = start + self.numAirlines
        state[-self.currentTradeSize:] = current_trade
        self.AirReplayMemory.add_record(next_state=state, action=air_action, reward=0, initial=True)
        air_action = self.pick_air_action(state, eps)
        current_trade[start:end] = air_action

        start = end
        end = start + self.numCombs
        state[-self.currentTradeSize:] = current_trade
        self.FlReplayMemory.add_record(next_state=state, action=fl_action, reward=0, initial=True)
        fl_action = self.pick_fl_action(state, eps)
        current_trade[start:end] = fl_action

        if not last_step:
            state[-self.currentTradeSize:] = current_trade
            self.AirReplayMemory.add_record(next_state=state, action=air_action, reward=0)
            self.FlReplayMemory.add_record(next_state=state, action=fl_action, reward=0)
            return current_trade
        else:
            state = torch.ones_like(state) * -1
            return current_trade, state, air_action, fl_action

    def assign_end_episode_reward(self, last_state, air_action, fl_action, shared_reward):
        self.AirReplayMemory.add_record(next_state=last_state, action=air_action, reward=shared_reward, done=1)
        self.FlReplayMemory.add_record(next_state=last_state, action=fl_action, reward=shared_reward, done=1)

    def train(self):
        air_batch = self.AirReplayMemory.sample(self.batchSize)
        fl_batch = self.FlReplayMemory.sample(self.batchSize)
        self.AirAgent.update_weights(air_batch)
        self.FlAgent.update_weights(fl_batch)
