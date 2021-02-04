import copy
from typing import List

import torch
from torch import optim

from Training.replayMemory import ReplayMemory


class HyperAgent:

    def __init__(self, air_agent, fl_agent, train_mode=False):
        self.FlAgent = fl_agent
        self.AirAgent = air_agent
        self.trainMode = train_mode

        self.AirReplayMemory = ReplayMemory(4)
        self.FlReplayMemory = ReplayMemory(10)

        self.airOptimizer = optim.Adam(self.AirAgent.parameters(), weight_decay=1e-5)
        self.flOptimizer = optim.Adam(self.FlAgent.parameters(), weight_decay=1e-5)

    def pick_air_action(self, state):
        scores = self.AirAgent.pick_action(state)
        action = torch.zeros_like(scores)
        action[torch.argmax(scores)] = 1
        return action

    def pick_fl_action(self, state):
        scores = self.FlAgent.pick_action(state)
        action = torch.zeros_like(scores)
        action[torch.argmax(scores)] = 1
        return action

    def step(self, state_list: List, last_step = False):
        current_trade = torch.zeros(28)
        state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)
        self.AirReplayMemory.set_initial_state(state)

        air_action = self.pick_air_action(state)
        current_trade[0:4] = air_action
        state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)

        self.FlReplayMemory.set_initial_state(state)
        fl_action = self.pick_fl_action(state)
        current_trade[4:14] = fl_action

        state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)
        self.AirReplayMemory.add_record(next_state=state, action=air_action, reward=0, initial=True)
        air_action = self.pick_air_action(state)
        current_trade[14:18] = air_action

        state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)
        self.FlReplayMemory.add_record(next_state=state, action=fl_action, reward=0, initial=True)
        fl_action = self.pick_fl_action(state)
        current_trade[18:28] = fl_action

        if not last_step:
            state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)
            self.AirReplayMemory.add_record(next_state=state, action=air_action, reward=0)
            self.FlReplayMemory.add_record(next_state=state, action=fl_action, reward=0)
            return current_trade
        else:
            state = torch.ones_like(state) * -1
            return current_trade, state, air_action, fl_action

    def assign_end_episode_reward(self, last_state, air_action, fl_action, shared_reward):
        self.AirReplayMemory.add_record(next_state=last_state, action=air_action, reward=shared_reward, done=1)
        self.FlReplayMemory.add_record(next_state=last_state, action=fl_action, reward=shared_reward, done=1)



