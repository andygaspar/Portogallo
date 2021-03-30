import copy
from typing import List

import numpy as np
import torch
from torch import optim
from Training.Agents.replayMemory import ReplayMemory
from Training.Agents import attentionAgent
from Training.masker import Masker


class AttentiveHyperAgent:

    def __init__(self, num_airlines, num_flights, num_trades, weight_decay, l_rate, discretisation_size=50,
                 trainings_per_step=10, batch_size=200, memory_size=10000, train_mode=False):

        MAX_NUM_FLIGHTS = 200
        self.singleTradeSize = num_flights  # 2 as we are dealing with couples
        self.currentTradeSize = self.singleTradeSize
        self.discretisationSize = discretisation_size

        self.weightDecay = weight_decay

        hidden_dim = 64

        self.network = attentionAgent.AttentionNet(hidden_dim, len_discretisation=self.discretisationSize,
                                                   l_rate=l_rate, weight_decay=weight_decay)

        self.trainMode = train_mode
        self.trainingsPerStep = trainings_per_step
        self.batchSize = batch_size

        self.replayMemory = ReplayMemory(MAX_NUM_FLIGHTS * self.discretisationSize, size=memory_size)

    def pick_flight(self, state, eps, masker: Masker, num_flights, num_airlines):
        actions = torch.zeros_like(masker.mask)
        if self.trainMode and np.random.rand() < eps:
            action = np.random.choice([i for i in range(len(masker.mask)) if round(masker.mask[i].item()) == 1])
            masker.set_action(action)
            actions[action] = 1
            return actions
        actions_tensor = state[:(self.discretisationSize+num_airlines) * num_flights]
        actions_tensor = actions_tensor.reshape((num_flights, self.discretisationSize + num_airlines))
        scores = torch.tensor(
            [self.network.pick_action(state, actions_tensor[i], masker.mask, num_flights, num_airlines).item()
             if masker.mask[i] == 1 else -float('inf') for i in range(actions_tensor.shape[0])
             ])
        action = torch.argmax(scores)
        actions[action] = 1
        masker.set_action(action.item())
        if not self.trainMode:
            print(scores)
        return actions

    def step(self, schedule: torch.tensor, eps, instance,
             len_step, masker=None, last_step=False, train=True):

        num_flights = instance.numFlights
        current_trade = torch.zeros(num_flights)
        state = torch.cat([schedule, current_trade], dim=-1)
        masker.set_initial_mask()

        if masker.mask is None:
            return None, torch.ones_like(state) * -1, None

        self.replayMemory.set_initial_state(state, masker.mask)

        for _ in range(len_step - 1):
            mask = masker.mask.clone()
            action = self.pick_flight(state, eps, masker, instance.numFlights, instance.numAirlines)
            current_trade += action
            state[-num_flights:] = current_trade
            self.replayMemory.add_record(next_state=state, action=action, mask=mask, reward=0)

        action = self.pick_flight(state, eps, masker, instance.numFlights, instance.numAirlines)
        current_trade += action
        state[-num_flights:] = current_trade

        if not last_step:
            state[-num_flights:] = current_trade
            self.replayMemory.add_record(next_state=state, action=action, mask=masker.mask, reward=0)
            return current_trade, None, None
        else:
            last_state = torch.ones_like(state) * -1
            return current_trade, last_state, action

    def assign_end_episode_reward(self, last_state, action, mask, shared_reward, actions_in_episode):
        self.replayMemory.add_record(next_state=last_state, action=action, mask=mask,
                                     reward=shared_reward, actions_in_episode=actions_in_episode, final=True)

    def assign_shorter_episode_reward(self, reward, instance_size, actions_in_episode):
        self.replayMemory.end_short_episode(reward, instance_size, actions_in_episode)

    def train(self):
        for i in range(self.trainingsPerStep):
            batch, idxs = self.replayMemory.sample(self.batchSize)
            loss = self.network.update_weights(batch)
            self.replayMemory.update_losses(idxs, loss)

    def episode_training(self, num_actions):
        batch = self.replayMemory.get_last_episode(num_actions)
        self.network.update_weights_episode(batch)

    def compute_reward(self, instance):
        shared_reward = -1000 * \
                        (0.08 - (instance.initialTotalCosts - instance.compute_costs(instance.flights, which="final"))
                         / instance.initialTotalCosts) / 0.08
