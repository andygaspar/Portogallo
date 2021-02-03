import copy
from typing import List
from Training import instanceMaker
from Training.replayMemory import ReplayMemory

import torch


class Trainer:

    def __init__(self, air_agent, fl_agent, length_episode):
        self.FlAgent = fl_agent
        self.AirAgent = air_agent
        self.lengthEpisode = length_episode

        self.AirReplayMemory = ReplayMemory()
        self.FlReplayMemory = ReplayMemory()

    def pick_action(self, scores):
        action = torch.zeros_like(scores)
        action[torch.argmax(scores)] = 1
        return action

    def step(self, state_list: List):
        current_trade = torch.zeros(28)
        state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)
        self.AirReplayMemory.set_initial_state(state)

        air_action = self.pick_action(self.AirAgent(state))
        current_trade[0:4] = air_action
        state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)

        self.FlReplayMemory.set_initial_state(state)
        fl_action = self.pick_action(self.FlAgent(state))
        current_trade[4:14] = fl_action

        self.AirReplayMemory.add_record(next_state=state, action=air_action, reward=0, initial=True)
        state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)
        air_action = self.pick_action(self.AirAgent(state))
        current_trade[14:18] = self.pick_action(air_action)

        self.FlReplayMemory.add_record(next_state=state, action=fl_action, reward=0, initial=True)
        state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)
        fl_action = self.pick_action(self.FlAgent(state))
        current_trade[18:28] = self.pick_action(fl_action)

        return current_trade

    def episode(self, schedule_tensor: torch.tensor):
        trade_list = torch.zeros(28 * self.lengthEpisode)
        for i in range(self.lengthEpisode):
            trade_list[i * 28: (i + 1) * 28] = self.step([schedule_tensor, trade_list])
        return trade_list

    def run(self, num_iterations, df=None):
        if df is not None:
            instance = instanceMaker.Instance(triples=False, df=df)
        else:
            #la creazione dell'istanza dovrà essere messa nel for successivamente
            return

        for i in range(num_iterations):
            schedule = instance.get_schedule_tensor()
            num_flights = instance.numFlights
            num_airlines = instance.numAirlines
            actions = self.episode(schedule)

    def compute_air_reward(self):
        pass