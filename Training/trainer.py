import copy
from typing import List
from Training import instanceMaker
from Training.replayMemory import ReplayMemory
from Training.airAgent import AirNet

import torch
from torch import nn, optim

import xpress as xp


class Trainer:

    def __init__(self, air_agent: AirNet, fl_agent, hyper_agent, length_episode):
        self.hyperAgent = hyper_agent
        self.FlAgent = fl_agent
        self.AirAgent = air_agent
        self.lengthEpisode = length_episode

        self.AirReplayMemory = ReplayMemory(4)
        self.FlReplayMemory = ReplayMemory(10)

        self.airOptimizer = optim.Adam(self.AirAgent.parameters(), weight_decay=1e-5)
        self.flOptimizer = optim.Adam(self.FlAgent.parameters(), weight_decay=1e-5)

    def train(self):
        air_batch = self.AirReplayMemory.sample(200)
        fl_batch = self.FlReplayMemory.sample(200)
        self.AirAgent.update_weights(air_batch)
        self.FlAgent.update_weights(fl_batch)

    # def step(self, state_list: List, last_step = False):
    #     current_trade = torch.zeros(28)
    #     state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)
    #     self.AirReplayMemory.set_initial_state(state)
    #
    #     air_action = self.AirAgent.pick_action(state)
    #     current_trade[0:4] = air_action
    #     state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)
    #
    #     self.FlReplayMemory.set_initial_state(state)
    #     fl_action = self.FlAgent.pick_action(state)
    #     current_trade[4:14] = fl_action
    #
    #     state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)
    #     self.AirReplayMemory.add_record(next_state=state, action=air_action, reward=0, initial=True)
    #     air_action = self.AirAgent.pick_action(state)
    #     current_trade[14:18] = air_action
    #
    #     state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)
    #     self.FlReplayMemory.add_record(next_state=state, action=fl_action, reward=0, initial=True)
    #     fl_action = self.FlAgent.pick_action(state)
    #     current_trade[18:28] = fl_action
    #
    #     if not last_step:
    #         state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)
    #         self.AirReplayMemory.add_record(next_state=state, action=air_action, reward=0)
    #         self.FlReplayMemory.add_record(next_state=state, action=fl_action, reward=0)
    #         return current_trade
    #     else:
    #         state = torch.ones_like(state) * -1
    #         return current_trade, state, air_action, fl_action

    def episode(self, schedule_tensor: torch.tensor, instance):
        trade_list = torch.zeros(28 * self.lengthEpisode)
        for i in range(self.lengthEpisode):
            trades = self.hyperAgent.step([schedule_tensor, trade_list])
            trade_list[i * 28: (i + 1) * 28] = trades

        trades, last_state, air_action, fl_action = self.hyperAgent.step([schedule_tensor, trade_list], last_step=True)
        instance.set_matches(trade_list, self.lengthEpisode, 28)

        instance.run()

        # instance.print_performance()
        shared_reward = - instance.compute_costs(instance.flights, which="final")
        self.hyperAgent.assign_end_episode_reward(last_state, air_action, fl_action, shared_reward)
        # self.AirReplayMemory.add_record(next_state=last_state, action=air_action, reward=shared_reward, done=1)
        # self.FlReplayMemory.add_record(next_state=last_state, action=fl_action, reward=shared_reward, done=1)

    def run(self, num_iterations, df=None):
        xp_problem= xp.problem()
        for i in range(num_iterations):
            print(i)

            instance = instanceMaker.Instance(triples=False, df=df, xp_problem=xp_problem)
            schedule = instance.get_schedule_tensor()
            num_flights = instance.numFlights
            num_airlines = instance.numAirlines
            self.episode(schedule, instance)
            if i >= 100:
                self.train()

            if i >= 100 and i % 10 == 0:
                instance.run()
                print(instance.matches)
                instance.print_performance()


    def compute_air_reward(self):
        pass