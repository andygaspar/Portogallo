import copy
from typing import List

import numpy as np

from Training import instanceMaker
from Training.replayMemory import ReplayMemory
from Training.airAgent import AirNet

import torch
from torch import nn, optim

import xpress as xp


class Trainer:

    def __init__(self, hyper_agent, length_episode, eps_decay=100):
        self.hyperAgent = hyper_agent
        self.lengthEpisode = length_episode
        self.eps = 1
        self.epsDecay = eps_decay # not used yet

    def episode(self, schedule_tensor: torch.tensor, instance, eps):
        trade_list = torch.zeros(28 * self.lengthEpisode)
        for i in range(self.lengthEpisode):
            trades = self.hyperAgent.step([schedule_tensor, trade_list], eps)
            trade_list[i * 28: (i + 1) * 28] = trades

        trades, last_state, air_action, fl_action = self.hyperAgent.step([schedule_tensor, trade_list], eps, last_step=True)
        instance.set_matches(trade_list, self.lengthEpisode, 28)

        instance.run()
        # instance.print_performance()
        shared_reward = - instance.compute_costs(instance.flights, which="final")
        self.hyperAgent.assign_end_episode_reward(last_state, air_action, fl_action, shared_reward)


    def run(self, num_iterations, df=None):
        xp_problem = xp.problem()
        for i in range(num_iterations):
            print(i)

            instance = instanceMaker.Instance(triples=False, df=df, xp_problem=xp_problem)
            schedule = instance.get_schedule_tensor()
            num_flights = instance.numFlights
            num_airlines = instance.numAirlines
            self.eps = np.exp(- 4*i/num_iterations)
            self.episode(schedule, instance, self.eps)
            if i >= 100:
                self.hyperAgent.train()

            if i >= 100 and i % 25 == 0:
                instance.run()
                print(instance.matches)
                instance.print_performance()

# to do
    def compute_air_reward(self):
        pass