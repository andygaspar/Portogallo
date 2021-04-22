import copy
from typing import List, Union

import numpy as np
import math as mt

from Training import instanceMaker
from Training.Agents.hyperAttentiveAgent import AttentiveHyperAgent
from Training.Agents.replayMemory import ReplayMemory
from Agents.attention import attentionFucker

import torch
from torch import nn, optim

import xpress as xp

from Training.masker import Masker


class Trainer:

    def __init__(self, hyper_agent: Union[AttentiveHyperAgent, attentionFucker.AttentionFucker],
                 length_episode, eps_fun, min_reward=-1000,
                 eps_decay=100, masker=Masker, triples=False):
        self.hyperAgent = hyper_agent
        self.lengthEpisode = length_episode
        self.eps = 1
        self.epsDecay = eps_decay  # not used yet
        self.epsFun = eps_fun
        self.triples = triples
        self.lenStep = 4 if not self.triples else 6
        self.actionsInEpisodes = 4 if not self.triples else 6

        self.Masker = masker

        self.k = min_reward
        self.a = np.exp(self.k) / (1 - np.exp(self.k))
        self.b = -np.log(1 / (1 - np.exp(self.k)))

    def episode(self, schedule_tensor: torch.tensor, instance, eps):
        masker = self.Masker(instance, self.triples)

        last_state, actions = self.hyperAgent.step(schedule_tensor, eps, instance,
                                                        len_step=self.lenStep, masker=masker, last_step= False, train= True)

        flight_trade_idx = masker.actions
        instance.set_matches(flight_trade_idx, len(flight_trade_idx) // self.lenStep, self.triples)
        instance.reset_and_run()
        # instance.print_performance()

        # shared_reward = 100 -100 * \
        #                 (0.08 - (instance.initialTotalCosts - instance.compute_costs(instance.flights, which="final"))
        #                  / instance.initialTotalCosts) / 0.08
        shared_reward = (instance.initialTotalCosts - instance.compute_costs(instance.flights, which="final"))*1000/ instance.initialTotalCosts
        if not self.hyperAgent.trainMode:
            print("reward", shared_reward)

        self.hyperAgent.assign_end_episode_reward(last_state, actions, masker.mask, shared_reward)
        self.hyperAgent.episode_training()

    def test_episode(self, schedule_tensor: torch.tensor, instance, eps):
        self.hyperAgent.trainMode = False
        self.episode(schedule_tensor, instance, eps)
        self.hyperAgent.trainMode = True

    def run(self, num_iterations, df=None, training_start_iteration=100, train_t=200):
        xp_problem = xp.problem()

        N = num_iterations // 1000
        idx = 0
        instance = None
        schedule = None
        instance = instanceMaker.Instance(triples=False, df=df, xp_problem=xp_problem)
        for k in range(N):
            for j in range(10):
                for i in range(100):
                    schedule = instance.get_schedule_tensor()
                    self.eps = self.epsFun(idx, num_iterations)
                    self.episode(schedule, instance, self.eps)
                    idx += 1
                #print("{0} {1:2f} {2:2f}".format(idx, self.hyperAgent.loss))
                print(idx, self.hyperAgent.loss, self.eps)

            print("\n TEST")
            print(instance.flights)
            self.test_episode(schedule, instance, self.eps)
            print(instance.matches)
            instance.print_performance()
                # print("{0} {1:2f} {2:2f} {3:4f}".format(i, self.hyperAgent.AirAgent.loss * s,
                #                                         self.hyperAgent.FlAgent.loss * s, self.eps))
