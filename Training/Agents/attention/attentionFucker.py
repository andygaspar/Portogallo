import copy
from typing import List

import numpy as np
import torch
from torch import optim
from Training.Agents.attention.attentionCodec import AttentionCodec
from Training.Agents import attentionAgent
from Training.Agents.attention.simpleNetTest import AgentNetwork
from Training.Agents.replayMemory import ReplayMemory
from Training.masker import Masker

sMax = torch.nn.Softmax(dim=-1)


class AttentionFucker:

    def __init__(self, num_airlines, num_flights, num_trades, discretisation_size, weight_decay, l_rate,
                 trainings_per_step=10, batch_size=200, memory_size=10000, train_mode=True):

        MAX_NUM_FLIGHTS = 200
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.singleTradeSize = num_flights  # 2 as we are dealing with couples
        self.currentTradeSize = self.singleTradeSize
        self.numAirlines = num_airlines
        self.discretisationSize = discretisation_size
        self.singleFlightSize = num_airlines + num_flights

        self.numFlights = num_flights
        self.weightDecay = weight_decay
        self.loss = 0

        hidden_dim = 64
        self._hidden_dim = hidden_dim
        self._context_dim = self._hidden_dim + self.numFlights
        self._codec = AttentionCodec(self.discretisationSize + self.numAirlines, self._hidden_dim, n_heads=2,
                                     n_attention_layers=2, context_dim=self._context_dim)
        self.context = None
        self.actions_embeddings = None

        self.network = AgentNetwork(weight_decay, num_flights)

        ##TRAINING DA VEDERE
        self.trainMode = train_mode
        self.trainingsPerStep = trainings_per_step
        self.batchSize = batch_size

        self.replayMemory = ReplayMemory(self.discretisationSize * MAX_NUM_FLIGHTS, self.discretisationSize,
                                               size=memory_size)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=l_rate)
        ######################

    def pick_flight(self, masker: Masker, state, eps):
        actions = torch.zeros_like(masker.mask)
        if self.trainMode and np.random.rand() < eps:
            action_idx = np.random.choice([i for i in range(len(masker.mask)) if round(masker.mask[i].item()) == 1])
            masker.set_action(action_idx)
            actions[action_idx] = 1
            return actions, action_idx
        with torch.no_grad():
            scores = self.network.forward(state)
        scores += torch.tensor([0 if el == 1 else -float("inf") for el in masker.mask]).to(self.device)
        action_idx = torch.argmax(scores)
        actions[action_idx] = 1
        masker.set_action(action_idx.item())

        if not self.trainMode:
            print(scores)

        return actions.to(self.device), action_idx

    def step(self, schedule: torch.tensor, eps, instance,
             len_step, initial=False, masker=None, last_step=False, train=True):
        schedule = schedule.to(self.device)
        num_flights = instance.numFlights

        current_trade = torch.zeros((5, 2)).to(self.device)
        state = torch.cat([schedule.flatten(), current_trade.flatten()], dim=-1)
        masker.set_initial_mask()

        self.replayMemory.set_initial_state(state)

        for i in range(len_step - 1):
            actions, action_idx = self.pick_flight(masker, state, eps)
            current_trade[i] = torch.tensor([instance.flights[action_idx].slot.index,
                                             instance.flights[action_idx].cost])
            state[num_flights*2:] = current_trade.flatten()
            self.replayMemory.add_record(next_state=state, action=actions, mask=masker.mask, reward=0)

        actions, _ = self.pick_flight(masker, state, eps)
        self.replayMemory.add_record(next_state=state, action=actions, mask=masker.mask, reward=0)
        last_state = torch.ones_like(state) * -1

        return last_state, actions

    def assign_end_episode_reward(self, last_state, action, mask, shared_reward):
        self.replayMemory.add_record(next_state=last_state, action=action, mask=mask,
                                     reward=shared_reward, final=True)

    def episode_training(self):
        if self.replayMemory.idx >= self.batchSize:
            for i in range(1):
                batch = self.replayMemory.sample(self.batchSize)
                self.network.update_weights(batch)
                self.loss = self.network.loss
        # self.optimizer.zero_grad()


        # _, _, rewards, probs = self.replayMemory.get_last_episode()
        # # probs = probs[probs < 0.99]
        # loss = (rewards.detach()[-1]-46.68) * torch.log(probs.T)
        # loss = loss.sum()
        # print(loss.item(), probs, rewards.detach()[-1])
        # with torch.autograd.set_detect_anomaly(True):
        #     loss.backward()
        # self.optimizer.step()
        # self.loss = loss.item()


#solution: [[FA8, FA16],[FB10, FB13],[FC7, FC15]]]