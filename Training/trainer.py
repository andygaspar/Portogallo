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

    def step(self, state_list: List, last_step = False):
        current_trade = torch.zeros(28)
        state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)
        self.AirReplayMemory.set_initial_state(state)

        air_action = self.pick_action(self.AirAgent(state))
        current_trade[0:4] = air_action
        state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)

        self.FlReplayMemory.set_initial_state(state)
        fl_action = self.pick_action(self.FlAgent(state))
        current_trade[4:14] = fl_action

        state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)
        self.AirReplayMemory.add_record(next_state=state, action=air_action, reward=0, initial=True)
        air_action = self.pick_action(self.AirAgent(state))
        current_trade[14:18] = self.pick_action(air_action)

        state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)
        self.FlReplayMemory.add_record(next_state=state, action=fl_action, reward=0, initial=True)
        fl_action = self.pick_action(self.FlAgent(state))
        current_trade[18:28] = self.pick_action(fl_action)

        if not last_step:
            state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)
            self.AirReplayMemory.add_record(next_state=state, action=air_action, reward=0)
            self.FlReplayMemory.add_record(next_state=state, action=fl_action, reward=0)
            return current_trade
        else:
            state = torch.cat(torch.ones_like(state) * -1, dim=-1)
            return current_trade, state, air_action, fl_action

    def episode(self, schedule_tensor: torch.tensor, instance):
        trade_list = torch.zeros(28 * self.lengthEpisode)
        for i in range(self.lengthEpisode):
            trades = self.step([schedule_tensor, trade_list])
            trade_list[i * 28: (i + 1) * 28] = trades

        trades, last_state, air_action, fl_action = self.step([schedule_tensor, trade_list], last_step=True)
        instance.set_matches(trade_list, self.lengthEpisode)
        instance.run()
        shared_reward = instance.initialTotalCosts - instance.compute_costs(instance.flights, which="final")
        self.FlReplayMemory.add_record(next_state=last_state, action=air_action, reward=shared_reward)
        self.FlReplayMemory.add_record(next_state=last_state, action=fl_action, reward=shared_reward)

    def run(self, num_iterations, df=None):
        for i in range(num_iterations):
            instance = instanceMaker.Instance(triples=False, df=df)
            schedule = instance.get_schedule_tensor()
            num_flights = instance.numFlights
            num_airlines = instance.numAirlines
            self.episode(schedule, instance)

    def compute_air_reward(self):
        pass