import copy
from typing import List

import numpy as np
import torch
from torch import optim

from Training.instanceMaker import Instance
from Training.replayMemory import ReplayMemory
import flAgent
import airAgent
from OfferChecker import checkOffer


class HyperAgent:

    def __init__(self, num_flight_types, num_airlines, num_flights, num_trades, num_combs, weight_decay, batch_size=200,
                 memory_size=10000, train_mode=False):

        ETA_info_size = 1
        time_info_size = 1
        self.singleTradeSize = (num_airlines + num_combs) * 2  # 2 as we are dealing with couples
        self.currentTradeSize = self.singleTradeSize
        self.numCombs = num_combs
        self.numAirlines = num_airlines

        input_size = (num_flight_types + ETA_info_size + time_info_size + num_airlines) * num_flights + \
                     num_trades * self.singleTradeSize + self.currentTradeSize

        self.weightDecay = weight_decay

        self.AirAgent = airAgent.AirNet(input_size, self.weightDecay, num_flight_types, num_airlines, num_flights,
                                        num_trades, num_combs)
        self.FlAgent = flAgent.FlNet(input_size, self.weightDecay, num_flight_types, num_airlines, num_flights,
                                     num_trades, num_combs)

        self.trainMode = train_mode
        self.batchSize = batch_size

        self.AirReplayMemory = ReplayMemory(self.numAirlines, input_size, size=memory_size)
        self.FlReplayMemory = ReplayMemory(self.numCombs, input_size, size=memory_size)
        self.finalState = torch.ones(input_size) * (-1)

    def pick_air_action(self, state, eps):
        if self.trainMode and np.random.rand() < eps:
            return np.random.choice(range(self.numAirlines))
        scores = self.AirAgent.pick_action(state)
        action = torch.zeros_like(scores)
        action[torch.argmax(scores)] = 1
        return action

    def pick_fl_action(self, state, eps):
        if self.trainMode and np.random.rand() < eps:
            return np.random.choice(range(self.numCombs))
        scores = self.FlAgent.pick_action(state)
        action = torch.zeros_like(scores)
        action[torch.argmax(scores)] = 1
        return action

    def step(self, state_list: List, eps, instance: Instance, last_step=False):
        current_trade = torch.zeros(self.singleTradeSize)
        state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)
        self.AirReplayMemory.set_initial_state(state)

        start = 0
        end = start + self.numAirlines
        air_action = self.pick_air_action(state, eps)
        airline = instance.airlines[air_action]
        if not self.feasible_offers_for_airline(airline, instance):
            self.AirReplayMemory.add_record(next_state=self.finalState, action=air_action,
                                            reward=-instance.initialTotalCosts, done=1)
            return [], False
        current_trade[start:end] = air_action
        state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)

        start = end
        end = start + self.numCombs
        self.FlReplayMemory.set_initial_state(state)
        fl_action = self.pick_fl_action(state, eps)
        couple = airline.flight_pairs[fl_action]
        if not self.feasible_couple(couple, instance):
            self.FlReplayMemory.add_record(next_state=self.finalState, action=air_action,
                                           reward=-instance.initialTotalCosts, done=1)
            return [], False
        current_trade[start:end] = fl_action

        start = end
        end = start + self.numAirlines
        state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)
        self.AirReplayMemory.add_record(next_state=state, action=air_action, reward=0, initial=True)
        air_action = self.pick_air_action(state, eps)
        airline_2 = instance.airlines[air_action]
        if airline.name == airline_2.name or not self.feasible_couple_for_airline(couple, airline_2, instance):
            self.AirReplayMemory.add_record(next_state=self.finalState, action=air_action,
                                            reward=-instance.initialTotalCosts, done=1)
            return [], False
        current_trade[start:end] = air_action

        start = end
        end = start + self.numCombs
        state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)
        self.FlReplayMemory.add_record(next_state=state, action=fl_action, reward=0, initial=True)
        fl_action = self.pick_fl_action(state, eps)
        couple_2 = airline_2.flight_pairs[fl_action]
        if not self.feasible_trade(couple, couple_2, instance):
            self.FlReplayMemory.add_record(next_state=self.finalState, action=air_action,
                                           reward=-instance.initialTotalCosts, done=1)
            return [], False
        current_trade[start:end] = fl_action

        if not last_step:
            state = torch.cat((state_list[0], state_list[1], current_trade), dim=-1)
            self.AirReplayMemory.add_record(next_state=state, action=air_action, reward=0)
            self.FlReplayMemory.add_record(next_state=state, action=fl_action, reward=0)
            return current_trade, True
        else:
            state = torch.ones_like(state) * (-1)
            return current_trade, state, air_action, fl_action

    def assign_end_episode_reward(self, last_state, air_action, fl_action, shared_reward):
        self.AirReplayMemory.add_record(next_state=last_state, action=air_action, reward=shared_reward, done=1)
        self.FlReplayMemory.add_record(next_state=last_state, action=fl_action, reward=shared_reward, done=1)

    def train(self):
        air_batch = self.AirReplayMemory.sample(200)
        fl_batch = self.FlReplayMemory.sample(200)
        self.AirAgent.update_weights(air_batch)
        self.FlAgent.update_weights(fl_batch)

    @staticmethod
    def feasible_offers_for_airline(airline, instance):
        if len(instance.offerChecker.check_airline_feasibility(airline, instance.airlines_pairs)) > 0:
            return True
        return False

    @staticmethod
    def feasible_couple(couple, instance):
        if len(instance.offerChecker.check_couple_in_pairs(couple, instance.airlines_pairs)) > 0:
            return True
        return False

    @staticmethod
    def feasible_couple_for_airline(couple, airline_2, instance):
        if len(instance.offerChecker.check_air_air_couple_match(couple, airline_2)) > 0:
            return True
        return False

    @staticmethod
    def feasible_trade(couple, couple_2, instance):
        if instance.offerChecker.check_trade(couple, couple_2):
            return True
        return False
