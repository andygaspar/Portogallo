import copy
from typing import List, Union

import numpy as np
import math as mt

from Training import instanceMaker
from Training.Agents.hyperAttentiveAgent import AttentiveHyperAgent
from Training.Agents.replayMemory import ReplayMemory
from Training.Agents.airAgent import AirNet
from Training.Agents.hyperAgent import HyperAgent

import torch
from torch import nn, optim

import xpress as xp

from Training.masker import Masker


class Trainer:

    def __init__(self, hyper_agent: Union[HyperAgent, AttentiveHyperAgent], length_episode, eps_fun, min_reward=-1000,
                 eps_decay=100, masker=Masker):
        self.hyperAgent = hyper_agent
        self.lengthEpisode = length_episode
        self.eps = 1
        self.epsDecay = eps_decay  # not used yet
        self.epsFun = eps_fun

        self.Masker = masker

        self.k = min_reward
        self.a = np.exp(self.k) / (1 - np.exp(self.k))
        self.b = -np.log(1 / (1 - np.exp(self.k)))

    def episode(self, schedule_tensor: torch.tensor, instance, eps):
        masker = self.Masker(instance)
        num_flights = instance.numFlights
        trade_list = torch.zeros(num_flights * self.lengthEpisode)
        for i in range(self.lengthEpisode):
            trade = self.hyperAgent.step(schedule_tensor, trade_list, eps, masker=masker)
            trade_list[i * num_flights: (i + 1) * num_flights] = trade

        trades, last_state, air_action, fl_action = self.hyperAgent.step(schedule_tensor, trade_list,
                                                                         eps, masker=masker, last_step=True)
        instance.set_matches(trade_list, self.lengthEpisode, num_flights)
        instance.run()
        instance.print_performance()

        print(instance.compute_costs(instance.flights, which="final"), instance.initialTotalCosts)

        shared_reward = mt.log(1 - instance.compute_costs(instance.flights, which="final") /
                               instance.initialTotalCosts + self.a) + self.b

        print(shared_reward)
        # shared_reward = -10000 * (instance.compute_costs(instance.flights, which="final")/instance.initialTotalCosts)
        self.hyperAgent.assign_end_episode_reward(last_state, air_action, fl_action,
                                                  masker.airMask, masker.flMask, shared_reward)

    def test_episode(self, schedule_tensor: torch.tensor, instance, eps):
        self.hyperAgent.trainMode = False
        self.episode(schedule_tensor, instance, eps)
        self.hyperAgent.trainMode = True

    def run(self, num_iterations, df=None, training_start_iteration=100, train_t=200):
        xp_problem = xp.problem()
        for i in range(training_start_iteration):
            print(i)
            instance = instanceMaker.Instance(triples=False, df=df, xp_problem=xp_problem)
            schedule = instance.get_schedule_tensor()

            num_flights = instance.numFlights
            num_airlines = instance.numAirlines
            self.episode(schedule, instance, eps=1)

        print('Finished initial exploration')
        self.hyperAgent.train()

        s = 10_000
        for i in range(training_start_iteration, num_iterations):
            instance = instanceMaker.Instance(triples=False, df=df, xp_problem=xp_problem)
            schedule = instance.get_schedule_tensor()
            num_flights = instance.numFlights
            num_airlines = instance.numAirlines
            self.eps = self.epsFun(i, num_iterations)
            self.episode(schedule, instance, self.eps)
            print("{0} {1:2f} {2:2f} {3:4f}".format(i, self.hyperAgent.AirAgent.loss * s,
                                                    self.hyperAgent.FlAgent.loss * s, self.eps))
            self.hyperAgent.train()
            if i % train_t == 0:
                self.hyperAgent.train()

                self.test_episode(schedule, instance, self.eps)
                print(instance.matches)
                instance.print_performance()
                # print("{0} {1:2f} {2:2f} {3:4f}".format(i, self.hyperAgent.AirAgent.loss * s,
                #                                         self.hyperAgent.FlAgent.loss * s, self.eps))

