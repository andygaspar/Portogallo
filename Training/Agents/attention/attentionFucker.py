import copy
from typing import List

import numpy as np
import torch
from torch import optim
from Training.Agents.attention.attentionCodec import AttentionCodec
from Training.Agents import attentionAgent
from Training.Agents.replayMemory import ReplayMemory, ReplayMemoryFucker
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

        ##TRAINING DA VEDERE
        self.trainMode = train_mode
        self.trainingsPerStep = trainings_per_step
        self.batchSize = batch_size

        self.replayMemory = ReplayMemoryFucker(self.discretisationSize * MAX_NUM_FLIGHTS, self.discretisationSize,
                                               size=memory_size)

        self.optimizer = torch.optim.Adam(self._codec.parameters(), lr=l_rate)
        ######################

    def pick_flight(self, masker: Masker, state):
        actions = torch.zeros_like(masker.mask)
        probs = self._codec.get_action_probs(state, self.actions_embeddings, masker.mask)
        if self.trainMode:
            action = torch.multinomial(probs.squeeze(), 1)
        else:
            action = torch.argmax(probs)

        actions[action] = 1
        masker.set_action(action.item())
        return actions.to(self.device), probs[action]

    def init_embeddings(self, schedule, num_flights):
        schedule = schedule.reshape((num_flights, -1)).to(self.device)
        self.actions_embeddings = self._codec.encode(schedule)
        self.context = torch.mean(self.actions_embeddings, dim=0).unsqueeze(0).to(self.device)
        #self.context = torch.zeros(1, self._context_dim)
        #self.context[0, :self._hidden_dim] = torch.mean(self.actions_embeddings, dim=0)

    def step(self, schedule: torch.tensor, eps, instance,
             len_step, initial=False, masker=None, last_step=False, train=True):
        self.init_embeddings(schedule, instance.numFlights)
        schedule = schedule.to(self.device)
        num_flights = instance.numFlights
        if initial:
            self.init_embeddings(schedule, num_flights)

        current_trade = torch.zeros(num_flights).to(self.device)
        c_trades = current_trade.to(self.device)
        state = torch.cat([schedule, current_trade], dim=-1)
        masker.set_initial_mask()

        self.replayMemory.set_initial_state(state)

        for _ in range(len_step - 1):
            action, act_prob = self.pick_flight(masker, torch.cat([self.context, c_trades.reshape((1, -1))], dim=-1))
            current_trade += action
            state[-num_flights:] = current_trade
            self.replayMemory.add_record(next_state=state, action=action, mask=masker.mask, reward=0, prob=act_prob)

        action, act_prob = self.pick_flight(masker, torch.cat([self.context, c_trades.reshape((1, -1))], dim=-1))
        current_trade += action
        state[-num_flights:] = current_trade

        last_state = state
        return current_trade, last_state, action, act_prob

    def assign_end_episode_reward(self, last_state, action, prob, mask, shared_reward, actions_in_episode):
        self.replayMemory.add_record(next_state=last_state, action=action, mask=mask,
                                     reward=shared_reward, prob=prob, actions_in_episode=actions_in_episode, final=True)

    def episode_training(self):
        self.optimizer.zero_grad()

        _, _, rewards, probs = self.replayMemory.get_last_episode()
        # probs = probs[probs < 0.99]
        loss = (rewards.detach()[-1]-46.68) * torch.log(probs.T)
        loss = loss.sum()
        print(loss.item(), probs, rewards.detach()[-1])
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
        self.optimizer.step()
        self.loss = loss.item()

#solution: [[FA8, FA16],[FB10, FB13],[FC7, FC15]]]