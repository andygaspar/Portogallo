import copy
from typing import List

import numpy as np

from Training import instanceMaker
from Training.Agents.replayMemory import ReplayMemory
from Training.Agents.airAgent import AirNet

import torch
from torch import nn, optim

import xpress as xp


class Trainer:

    def __init__(self, hyper_agent, length_episode, eps_fun, eps_decay=100):
        self.hyperAgent = hyper_agent
        self.lengthEpisode = length_episode
        self.eps = 1
        self.epsDecay = eps_decay # not used yet
        self.epsFun = eps_fun

    def episode(self, schedule_tensor: torch.tensor, instance, eps):
        trade_size = self.hyperAgent.singleTradeSize
        trade_list = torch.zeros(trade_size * self.lengthEpisode)
        for i in range(self.lengthEpisode):
            trades = self.hyperAgent.step([schedule_tensor, trade_list], eps)
            trade_list[i * trade_size: (i + 1) * trade_size] = trades

        trades, last_state, air_action, fl_action = self.hyperAgent.step([schedule_tensor, trade_list],
                                                                         eps, last_step=True)
        instance.set_matches(trade_list, self.lengthEpisode, trade_size)

        instance.run()
        # instance.print_performance()
        shared_reward = - instance.compute_costs(instance.flights, which="final")/instance.initialTotalCosts
        self.hyperAgent.assign_end_episode_reward(last_state, air_action, fl_action, shared_reward)

    def test_episode(self, schedule_tensor: torch.tensor, instance, eps):
        self.hyperAgent.trainMode = False
        self.episode(schedule_tensor, instance, eps)
        self.hyperAgent.trainMode = True

    def run(self, num_iterations, df=None, training_start_iteration=100):
        xp_problem = xp.problem()
        for i in range(training_start_iteration):
            print(i)
            instance = instanceMaker.Instance(triples=False, df=df, xp_problem=xp_problem)
            schedule = instance.get_schedule_tensor()
            num_flights = instance.numFlights
            num_airlines = instance.numAirlines
            self.episode(schedule, instance, eps=1)

        for i in range(training_start_iteration, num_iterations):
            s = 10_000
            print("{0} {1:2f} {2:2f} {3:4f}".format(i, self.hyperAgent.AirAgent.loss*s,
                                                    self.hyperAgent.FlAgent.loss*s, self.eps))
            instance = instanceMaker.Instance(triples=False, df=df, xp_problem=xp_problem)
            schedule = instance.get_schedule_tensor()
            num_flights = instance.numFlights
            num_airlines = instance.numAirlines
            self.eps = self.epsFun(i, num_iterations)
            self.episode(schedule, instance, self.eps)
            self.hyperAgent.train()

            if i >= 100 and i % 25 == 0:
                self.test_episode(schedule, instance, self.eps)
                print(instance.matches)
                instance.print_performance()



# to do
    def compute_air_reward(self):
        pass