import copy
from typing import List, Union

import numpy as np
import math as mt

from Training import instanceMaker
from Training.Agents.hyperAttentiveAgent import AttentiveHyperAgent
from Training.Agents.replayMemory import ReplayMemory

import torch
from torch import nn, optim

import xpress as xp

from Training.masker import Masker


class Trainer:

    def __init__(self, hyper_agent: Union[AttentiveHyperAgent], length_episode, eps_fun, min_reward=-1000,
                 eps_decay=100, masker=Masker, triples=False):
        self.hyperAgent = hyper_agent
        self.lengthEpisode = length_episode
        self.eps = 1
        self.epsDecay = eps_decay  # not used yet
        self.epsFun = eps_fun
        self.triples = triples
        self.lenStep = 4 if not self.triples else 6
        self.actionsInEpisodes = 4 * self.lengthEpisode if not self.triples else 6 * self.lengthEpisode

        self.Masker = masker

        self.k = min_reward
        self.a = np.exp(self.k) / (1 - np.exp(self.k))
        self.b = -np.log(1 / (1 - np.exp(self.k)))

    def episode(self, schedule_tensor: torch.tensor, instance, eps):
        masker = self.Masker(instance, self.triples)
        self.hyperAgent.replayMemory.init_episode(self.actionsInEpisodes, instance.numFlights, instance.numAirlines,
                                                  self.lengthEpisode)
        trade, last_state, flight_trade_idx = None, None, []
        for i in range(self.lengthEpisode - 1):
            trade, last_state, _ = self.hyperAgent.step(schedule_tensor, eps, instance,
                                         len_step=self.lenStep, masker=masker)
            if trade is not None:
                flight_trade_idx += masker.actions
            else:
                break

        if trade is not None:
            trade, last_state, action = self.hyperAgent.step(schedule_tensor, eps, instance,
                                                             len_step=self.lenStep, masker=masker, last_step=True)
            if trade is not None:
                flight_trade_idx += masker.actions

        instance.set_matches(flight_trade_idx, len(flight_trade_idx)//self.lenStep, self.triples)
        instance.run()
        # instance.print_performance()
        # print(instance.compute_costs(instance.flights, which="final"), instance.initialTotalCosts)
        # shared_reward = mt.log(1 - instance.compute_costs(instance.flights, which="final") /
        #                        instance.initialTotalCosts + self.a) + self.b
        # print("pippo", 1 - instance.compute_costs(instance.flights, which="final") / instance.initialTotalCosts)
        shared_reward = -1000 * \
                        (0.08 - (instance.initialTotalCosts - instance.compute_costs(instance.flights, which="final"))
                         / instance.initialTotalCosts) / 0.08
        # print(shared_reward)
        # print(action)
        if trade is not None:
            self.hyperAgent.assign_end_episode_reward(last_state, action, masker.mask, shared_reward,
                                                      self.actionsInEpisodes)
        else:
            self.hyperAgent.assign_shorter_episode_reward(shared_reward, schedule_tensor.shape[0], len(flight_trade_idx))
        self.hyperAgent.episode_training(len(flight_trade_idx))

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
            self.episode(schedule, instance, eps=1)

        print('Finished initial exploration')

        s = 10_000
        for i in range(training_start_iteration, num_iterations):
            instance = instanceMaker.Instance(triples=False, df=df, xp_problem=xp_problem)
            schedule = instance.get_schedule_tensor()
            self.eps = self.epsFun(i, num_iterations)
            self.episode(schedule, instance, self.eps)
            print("{0} {1:2f} {2:2f}".format(i, self.hyperAgent.network.loss, self.eps))
            if i % train_t == 0:
                print("\n TEST")
                self.test_episode(schedule, instance, self.eps)
                print(instance.matches)
                instance.print_performance()
                # print("{0} {1:2f} {2:2f} {3:4f}".format(i, self.hyperAgent.AirAgent.loss * s,
                #                                         self.hyperAgent.FlAgent.loss * s, self.eps))
